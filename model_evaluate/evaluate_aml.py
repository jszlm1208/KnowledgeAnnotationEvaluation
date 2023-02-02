import argparse
import os
import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report

# import self defined modules
from data_preprocessing.data_generator_v1 import DataGenerator
from get_token_tag_v1 import GetTokenTag

def build_parse_args():
    parser = argparse.ArgumentParser(description='evaluate the model')
    parser.add_argument('--model_path', type=str, default='model_path', help='model path')
    parser.add_argument('--id2query_path', type=str, default='test_data/cjo_id2query.json', help='path to the id2query.json file')
    parser.add_argument('--labeled_file_path', type=str, default='test_data/cjo_labeledDatapoints.jsonl', help='path to the labeled file')
    parser.add_argument('--tokenizer_type', type=str, default='local', help='tokenizer type, bert or nltk or local')
    parser.add_argument('--output_path', type=str, default='output_data', help='save the result to output_path')
    return parser.parse_args()



def main(): 
    # parse the arguments
    args = build_parse_args()
    # print the arguments
    print(args)

    # 1. load data from AML data labeling tool
    data = DataGenerator(args.id2query_path, args.labeled_file_path, "labeled").data
    raw_datasets = GetTokenTag(data, 'labeled', args.tokenizer_type, args.model_path).token_tag
    #df_labeled = pd.DataFrame(labeled)
    #df_labeled.to_csv('output_data/labeled.csv', index=False)


    # 2. get model from pretrained
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)

    id2label = model.config.id2label
    label2id = {v: k for k, v in id2label.items()}

    # 3. update raw_datasets to align with prediction requirement
    raw_datasets_updated = []
    for item in raw_datasets:
        tokens = item['token']
        label_ids = [label2id[label] for label in item['tag']]
        raw_datasets_updated.append({"tokens": tokens, "label_ids": label_ids})


    def align_labels_with_tokens(label_ids, word_ids):
        new_label_ids = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label_id = -100 if word_id is None else label_ids[word_id]
                new_label_ids.append(label_id)
            elif word_id is None:
                # Special token
                new_label_ids.append(-100)
            else:
                # Same word as previous token
                label_id = label_ids[word_id]
                label = id2label[label_id]
                # If the label is B-XXX we change it to I-XXX
                if label.startswith("B-"):
                    label = "I" + label[1:]
                    label_id = label2id[label]
                new_label_ids.append(label_id)

        return new_label_ids
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], padding=True, is_split_into_words=True)
        label_ids = examples["label_ids"]
        word_ids = tokenized_inputs.word_ids()
        tokenized_inputs["labels"] = align_labels_with_tokens(label_ids, word_ids) # use "labels" to align with the key defined in the model
        return tokenized_inputs
    
    # 4. get tokenized datasets for prediction
    tokenized_datasets = []
    output_datasets = raw_datasets.copy()
    for i, item in enumerate(raw_datasets_updated):
        tokenized_inputs = tokenize_and_align_labels(item)
        input_ids = tokenized_inputs["input_ids"]
        output_datasets[i]['tokens_new'] = [tokenizer.decode(input_ids[i]) for i in range(1,len(input_ids)-1)]
        tokenized_datasets.append(tokenized_inputs)


    
    # 5. get predictions
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    eval_dataloader = DataLoader(tokenized_datasets, collate_fn=data_collator, batch_size=1)
    ## need to make changes to align the id2label dictionary in evaluation data and in the model
    def postprocess(predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions


    lst_pred = []
    lst_label = []
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        true_predictions, true_labels = postprocess(predictions, labels)
        assert len(true_predictions) == len(true_labels), "the length differs between predictions and labels"
        lst_pred += (true_predictions)
        lst_label += (true_labels)
    
    # 6. add prediction result to output_datasets and save result
    def get_span(tokens,labels):
        spans = ['None']*len(tokens)
        span_start = 0
        span_end = 0
        spans_id = [-1]*len(tokens) #save spans_id to do span count
        for i,t in enumerate(tokens):
            sign = 1
            if labels[i].startswith("B-"):
                span_start = i
                span_end = span_start + 1
            elif labels[i].startswith("I-"):
                span_end += 1  
            else:
                sign = 0
            if sign == 1:
                spans[span_start:span_end] = [' '.join(tokens[span_start:span_end])]*(span_end - span_start)
                spans_id[span_start:span_end] = [span_start]*(span_end - span_start)
        return spans,spans_id
    #{'queryId':queryId, 'query':query, 'token':tokenized_query, 'tag':lst_tag, 'span':lst_span, 'spanId':lst_spanId}
    QueryId = []
    Tokens = []
    Tokens_Old = []
    Tags_Pred = []
    Tags_Labeled = []
    Spans_Pred = []
    Spans_Labeled = []
    SpansId_Pred =[]
    SpansId_Labeled =[]
    for i, dct in enumerate(output_datasets):
        token_old = dct['tokens']
        token = dct['tokens_new']
        # if the evaluation dataset is from converted conll format, the tokenizer may be different. If from raw data with offset position, token_old and token should be the same.
        l_delta = len(token) - len(token_old)
        if l_delta > 0:
            token_old += ['None']*l_delta # make the same length

        tag_pred = lst_pred[i]
        span_pred, spanId_pred = get_span(token,tag_pred)

        queryId = [dct['queryId']]*len(token)
        tag_labeled = dct['tag']
        span_labeled = dct['span']
        spanId_labeled = dct['spanId']

        QueryId += queryId
        Tokens += token
        Tokens_Old += token_old
        Tags_Pred += tag_pred
        Tags_Labeled += tag_labeled
        Spans_Pred += span_pred
        Spans_Labeled += span_labeled
        SpansId_Pred += spanId_pred
        SpansId_Labeled += spanId_labeled
    assert len(Tags_Pred) == len(Tags_Labeled), "Tags_Pred and Tags_Labeled should have same length"
    
    df = pd.DataFrame({'QueryId':QueryId, 'Tokens':Tokens,'Tokens_Old':Tokens_Old, 'Tags_Pred':Tags_Pred, 'Tags_Labeled':Tags_Labeled, 
                       'Spans_Pred':Spans_Pred,'Spans_Labeled':Spans_Labeled, 'SpansId_Pred':SpansId_Pred, 'SpansId_Labeled':SpansId_Labeled})
    
    df['OrderId'] = df.index
    output_file = os.path.join(args.output_path, 'data.csv')
    # check if output_path exists, if not, create it
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    df.to_csv(output_file, index=False)

    # 7. print the evaluation result

    print(classification_report(lst_label,lst_pred))


    

if __name__ == '__main__':
    main()
