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
import json

# import self defined modules
sys.path.append('../model_evaluate/data_preprocessing')
from data_generator_v1 import DataGenerator
from get_token_tag_v1 import GetTokenTag

sys.path.append('../utils')
from data_utils import get_span, get_id2query

def build_parse_args():
    parser = argparse.ArgumentParser(description='evaluate the model')
    parser.add_argument('--model_path', type=str, default='model_path', help='model path')
    parser.add_argument('--aml', type=bool, default=False, help='run the script in AML or local')
    parser.add_argument('--data_path_id2query', type=str, default='data_path_id2query', help='raw files dataset to get id2query')
    parser.add_argument('--data_path_labeled', type=str, default='data_path_labeled', help='exported labeled data')
    parser.add_argument('--tokenizer_type', type=str, default='local', help='tokenizer type, bert or nltk or local')
    parser.add_argument('--output_path', type=str, default='output_data', help='save the result to output_path')
    return parser.parse_args()

# read mapping file
def read_mappingfile(id2query_path):
    with open(id2query_path, 'r') as f:
        data = json.load(f)
    return data

def main(): 
    # parse the arguments
    args = build_parse_args()
    # print the arguments
    print(args)

    # 1. load data from AML data labeling tool
    filenames = next(os.walk(args.data_path_labeled))[2]
    if args.aml is True:
        id2query = get_id2query(args.data_path_id2query)
    else:
        id2query_file = [i for i in filenames if 'id2query' in i][0]
        id2query = read_mappingfile(os.path.join(args.data_path_labeled, id2query_file))
    
    labeled_file = [i for i in filenames if 'labeledDatapoints' in i][0]
    
    labeled_file_path = os.path.join(args.data_path_labeled, labeled_file)

    data = DataGenerator(id2query, labeled_file_path, "labeled").data
    raw_datasets = GetTokenTag(data, 'labeled', args.tokenizer_type, args.model_path).token_tag
    #df_labeled = pd.DataFrame(labeled)
    #df_labeled.to_csv('output_data/labeled.csv', index=False)
    #print(len(raw_datasets))
    
    # 2. get model from pretrained
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)

    id2label = model.config.id2label
    label2id = {v: k for k, v in id2label.items()}

    texts = [ dct['query'] for dct in raw_datasets]
    pt_batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**pt_batch).logits
    predictions = torch.argmax(logits, axis=-1)
    predictions = predictions.detach().cpu().clone().numpy()


    # 3. get the span and output the result as dataframe format
    lst_pred = []
    lst_label =[]
    for i, dct in enumerate(raw_datasets):
        label = dct['tag']
        pred_id = predictions[i]
        pred = [id2label[pred_id[i]] for i in range(1,len(label)+1)]
        lst_pred.append(pred)
        lst_label.append(label)
    
    output_datasets = raw_datasets.copy()
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
        token_old = dct['token']
        token = dct['token']
        tag_pred = lst_pred[i]
        span_pred, spanId_pred = get_span(token,tag_pred, tokens_type='subword')
        queryId = [dct['queryId']]*len(token)
        tag_labeled = dct['tag']
        if len(tag_labeled) != len(tag_pred):
            print(f"tag_labeled length diffs, QueryId: {i}")
        
        span_labeled, spanId_labeled = get_span(token, tag_labeled, tokens_type='subword')

        QueryId += queryId
        Tokens += token
        Tokens_Old += token_old
        Tags_Pred += tag_pred
        Tags_Labeled += tag_labeled
        Spans_Pred += span_pred
        Spans_Labeled += span_labeled
        SpansId_Pred += spanId_pred
        SpansId_Labeled += spanId_labeled
    
    
    df = pd.DataFrame({'QueryId':QueryId, 'Tokens':Tokens,'Tokens_Old':Tokens_Old, 'Tags_Pred':Tags_Pred, 'Tags_Labeled':Tags_Labeled, 
                       'Spans_Pred':Spans_Pred,'Spans_Labeled':Spans_Labeled, 'SpansId_Pred':SpansId_Pred, 'SpansId_Labeled':SpansId_Labeled})
    
    df['OrderId'] = df.index
    output_file = os.path.join(args.output_path, 'data.csv')
    # check if output_path exists, if not, create it
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    df.to_csv(output_file, index=False)

    # save id2query to json
    with open(os.path.join(args.output_path,'id2query.json'), 'w') as f:
        json.dump(id2query, f)

    # 4. print the evaluation result

    print(classification_report(lst_label,lst_pred))

    
    # 5. optional: check the prediction to see if any abnormal cases and save the query to json file as test cases
    
    prediction_ab = df.query('Tags_Pred != "O" and Spans_Pred == "None"')
    lst_QueryId = [i for i in prediction_ab['QueryId']]
    lst_QueryId = list(set(lst_QueryId))
    print(f"QueryId with abnormal prediction: {lst_QueryId}")
    """
    with open('model_testcases.json','a') as f:
        for i in lst_QueryId:
            f.write(json.dumps({i:texts[int(i)]}))
            f.write('\n') 
    """

    

if __name__ == '__main__':
    main()
