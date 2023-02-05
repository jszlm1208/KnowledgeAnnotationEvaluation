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
from nltk.tokenize import word_tokenize
import json
import re


class DataGenerator:
    """"
    This class is used to convert data output by annotation and by labeling to unified format.
    id2query_path: file that maps query id to query, json file
    file_path: file that contains data output by annotation or by labeling, json file
    source: ['predicted', 'labeled'], if 'predicted', then file_path is the output of annotation, if 'labeled', then file_path is the output of labeling
    """
    def __init__(self, id2query_path,file_path, source=['predicted', 'labeled']):
        self.id2query = self._read_mappingfile(id2query_path)
       
        if source == 'predicted':
            self.data, self.lst_error = self._read_predictedfile(file_path)
        elif source == 'labeled':
            self.data, self.lst_error = self._read_labeledfile(file_path)
        else:
            raise ValueError('source should be either predicted or labeled')

        assert len(self.data) == len(self.id2query), "output file and id2query file should have same length"
       

    # read mapping file
    def _read_mappingfile(self, id2query_path):
        with open(id2query_path, 'r') as f:
            data = json.load(f)
        return data
    
     
    
    # read predicted file
    def _read_predictedfile(self, predicted_path):
        query2id = {v:k for k,v in self.id2query.items()}
        def _normalize(dict_predicted):
            dict_predicted_new = {}
            query = dict_predicted['Query']
            dict_predicted_new['queryId'] = int(query2id[query])
            dict_predicted_new['query'] = query
            lst_label = dict_predicted['MatchedEntityList']
            lst_label_new = []
            for dct in lst_label:
                dct_new = {}
                dct_new['label'] = dct['EntityType'].split('.')[1]
                dct_new['span'] = dct['MatchedText']
                dct_new['offsetStart'] = dct['Start']
                dct_new['offsetEnd'] = dct['Start'] + dct['Length']
                dct_new['type'] = 'by token'
                lst_label_new.append(dct_new)
            dict_predicted_new['label'] = lst_label_new
            return dict_predicted_new
        data = [{'queryId':key,'query':value} for key,value in self.id2query.items()]
        with open(predicted_path, 'r') as f:
            temp = json.load(f)
        # order the predicted results by queryId
        lst_error = []
        for dct in temp:
            try:
                dct_new = _normalize(dct)
                data[int(dct_new['queryId'])]['label'] = dct_new['label']
            except:
                lst_error.append(dct)
        return data, lst_error
    
    # read labeled file
    def _read_labeledfile(self, labeled_path):
        def _get_queryId(text):
            return text[(text.rfind('_')+1):text.rfind('.')]
        
        def _normalize(dict_labeled):
            dict_labeled_new = {}
            queryId = _get_queryId(dict_labeled['image_url'])
            query = self.id2query[queryId]
            dict_labeled_new['queryId'] = int(queryId)
            dict_labeled_new['query'] = query
            lst_label = dict_labeled['label']
            lst_label_new = []
            for dct in lst_label:
                dct_new = {}
                dct_new['label'] = dct['label']
                dct_new['span'] = query[dct['offsetStart']:dct['offsetEnd']]
                dct_new['offsetStart'] = dct['offsetStart']
                dct_new['offsetEnd'] = dct['offsetEnd']
                dct_new['type'] = 'by character'
                lst_label_new.append(dct_new)
            dict_labeled_new['label'] = lst_label_new
            return dict_labeled_new
        data = [{'queryId':key,'query':value, 'label':[]} for key,value in self.id2query.items()]
        lst_error = []
        with open(labeled_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                try:
                    temp = _normalize(json.loads(line))
                    data[int(temp['queryId'])]['label'] = temp['label'] # order the labeled results by queryId
                except:
                    lst_error.append(json.loads(line))
        return data, lst_error


class Tokenizer:
    # lst_char = ['.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '/', '\\', '"', "'", '-', '_', '+', '=', '*', '&', '^', '%', '$', '#', '@', '~', '`', '<', '>', '|', ' ']
    def __init__(self, text, tokenizer_type=['bert', 'nltk','local'], model_path=None):
        lst_char = ['"', "'"]
        pattern = '[' + ''.join(lst_char) + ']'
        self.tokenizer_type = tokenizer_type
        self.text = text
        self.model_path = model_path
        self.tokenized_text = self.tokenize_text(self.text, self.tokenizer_type, self.model_path)
        #self.tokenized_text = [re.sub(pattern, '', token) for token in tokenized_text]

    def tokenize_text(self, text, tokenizer_type, model_path):
        if tokenizer_type == 'bert':
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            tokenized_text = tokenizer.tokenize(text)
        elif tokenizer_type == 'nltk':
            tokenized_text = word_tokenize(text)
        #locally saved model
        elif tokenizer_type == 'local': 
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenized_text = tokenizer.tokenize(text)
        else:
            raise ValueError('Invalid tokenizer type')
        return tokenized_text

class GetTokenTag:
    """
    This class is used to get tokenized query and corresponding tag based on the output of annotation or labeling unified format by data_generator.py
    data can be generated by data = DataGenerator(id2query_path,file_path,source).data
    tokenizer: tokenizer function
    """
    def __init__(self, data, source=['predicted', 'labeled'], tokenizer_type=['bert', 'nltk','local'], model_path=None):
        self.data = data
        self.source = source
        self.tokenizer_type = tokenizer_type
        self.model_path = model_path
        if source == 'predicted':
            self.token_tag = self._get_token_tag_predicted(self.data, self.tokenizer_type, self.model_path)
        elif source == 'labeled':
            self.token_tag = self._get_token_tag_labeled(self.data, self.tokenizer_type, self.model_path)
        else:
            raise ValueError('Invalid source')

    def _get_token_tag_labeled(self, data, tokenizer_type, model_path):
        data_token_tag = []
        for i, dct in enumerate(data):
            queryId = dct['queryId']
            query = dct['query']
            tokenized_query = Tokenizer(query, tokenizer_type, model_path).tokenized_text
            lst_label = dct['label']
            l = len(lst_label)
            if l > 0:
                lst_tag = []
                lst_span = []
                lst_spanId = []
                offset_dict = {}
                for j in range(l):
                    offsetStart = lst_label[j]['offsetStart']
                    offsetEnd = lst_label[j]['offsetEnd']
                    label = lst_label[j]['label']
                    offset_dict[int(offsetStart)] = [int(offsetEnd), label]
                offset_dict_keys = list(offset_dict.keys())
                offset_dict_keys.sort()
                start_index = 0
                for k, offset in enumerate(offset_dict_keys):
                    for item in Tokenizer(query[start_index:offset], tokenizer_type, model_path).tokenized_text:
                        lst_tag.append('O')
                        lst_span.append('None')
                        lst_spanId.append(-1)
                    tokens = Tokenizer(query[offset:offset_dict[offset][0]], tokenizer_type, model_path).tokenized_text
                    if len(tokens) > 1:
                        tags = ['B-' + offset_dict[offset][1]] + ['I-' + offset_dict[offset][1]] * (len(tokens) - 1)
                        tags[0] = 'B-' + offset_dict[offset][1]
                        lst_tag += tags
                        lst_span += [query[offset:offset_dict[offset][0]]] * len(tokens)
                        lst_spanId += [k] * len(tokens)
                    else:
                        lst_tag.append('B-' + offset_dict[offset][1])
                        lst_span.append(query[offset:offset_dict[offset][0]])
                        lst_spanId.append(k)
                    start_index = offset_dict[offset][0]
                lst_tag += ['O'] * (len(tokenized_query) - len(lst_tag))
                lst_span += ['None'] * (len(tokenized_query) - len(lst_span))
                lst_spanId += [-1] * (len(tokenized_query) - len(lst_spanId))
                # revert back if labeler incorrectly split the token
                if len(lst_tag) != len(tokenized_query):
                    lst_tag = ['O'] * len(tokenized_query)
                    lst_span = ['None'] * len(tokenized_query)
                    lst_spanId = [-1] * len(tokenized_query)
                dct_new = {'queryId': queryId, 'query': query, 'token': tokenized_query, 'tag': lst_tag, 'span': lst_span, 'spanId': lst_spanId}
            else:
                dct_new = {'queryId': queryId, 'query': query, 'token': tokenized_query, 'tag': ['O'] * len(tokenized_query), 'span': ['None'] * len(tokenized_query), 'spanId': [-1] * len(tokenized_query)}
            data_token_tag.append(dct_new)
        return data_token_tag
            
    def _get_token_tag_predicted(self, data, tokenizer_type, model_path):
        data_token_tag = []
        for i, dct in enumerate(data):
            queryId = dct['queryId']
            query = dct['query']
            tokenized_query = Tokenizer(query, tokenizer_type, model_path).tokenized_text
            lst_label = dct['label']
            lst_tag = ['O'] * len(tokenized_query)
            lst_span = ['None'] * len(tokenized_query)
            lst_spanId = [-1] * len(tokenized_query)
            for j, token in enumerate(tokenized_query):
                tags = [] # store the tags for each token in the list, one token may have multiple tags
                spans = [] # store the spans for each token in the list, one token may have multiple spans
                spanIds = [] 
                for k, dct_label in enumerate(lst_label):
                    span = dct_label['span']
                    offsetStart = dct_label['offsetStart']
                    offsetEnd = dct_label['offsetEnd']
                    tokenized_span = Tokenizer(span, tokenizer_type, model_path).tokenized_text
                    lst = [ po for po, ch in enumerate(tokenized_query) if ch == tokenized_span[0] ]
                    # update position if not matched
                    #join list of tokens to string and compare with span

                    if dct_label['span'].casefold() != ' '.join(tokenized_query[offsetStart:offsetEnd]).casefold():
                        try:
                            offsetStart = tokenized_query.index(tokenized_span[0])
                            offsetEnd = offsetStart + len(tokenized_span)
                        except:
                            print('Error: span not found in tokenized query in queryId: ', queryId)
                    range_span = range(offsetStart, offsetEnd)
                    if (token.casefold() == (tokenized_span[0]).casefold()) and (j in range_span):
                        tags.append('B-' + dct_label['label'])
                        spans.append(dct_label['span'])
                        spanIds.append(k)
                    elif (token.casefold() in [x.casefold() for x in tokenized_span]) and (j in range_span):
                        tags.append('I-' + dct_label['label'])
                        spans.append(dct_label['span'])
                        spanIds.append(k)
                    else:
                        pass
                if len(tags) > 0:
                    lst_tag[j] = tags
                    lst_span[j] = spans
                    lst_spanId[j] = spanIds
            dct_new = {'queryId':queryId, 'query':query, 'token':tokenized_query, 'tag':lst_tag, 'span':lst_span, 'spanId':lst_spanId}
            data_token_tag.append(dct_new)
        return data_token_tag

def get_span(tokens,labels):
    #use cases B-, I-, O-
    spans = ['None']*len(tokens)
    span_start = -1
    span_length = 0
    spans_id = [-1]*len(tokens) #save spans_id to do span count
    sign = 0
    l = len(tokens)
    for i in range(l):
        if labels[i][0:2] == "B-":
            if ((i-1) in range(l)) and span_length > 0:
                span = ''.join(tokens[span_start:(span_start+span_length)])
                span = re.sub("▁"," ",span)
                span = span.strip()
                spans[span_start:(span_start+span_length)] = [span]*(span_length)
                spans_id[span_start:(span_start+span_length)] = [span_start]*(span_length)
                span_start = i
                span_length = 1
            else:
                span_start = i
                span_length = 1
        elif labels[i][0:2] == "I-":
            if ((i-1) in range(l)):
                if (labels[i-1][0:2] == "B-") and (labels[i-1][2:] == labels[i][2:]):
                    span_length += 1
                elif (labels[i-1][0:2] == "I-") and (span_length > 1):
                    span_length += 1
                else:
                    span_length = 0
            else:
                span_length = 0
        else:
            if span_length > 0:
                span = ''.join(tokens[span_start:(span_start+span_length)])
                span = re.sub("▁"," ",span)
                span = span.strip()
                spans[span_start:(span_start+span_length)] = [span]*(span_length)
                spans_id[span_start:(span_start+span_length)] = [span_start]*(span_length)
                span_length = 0                   
    #check the end tag
    if span_length > 0:
        span = ''.join(tokens[span_start:(span_start+span_length)])
        span = re.sub("▁"," ",span)
        span = span.strip()
        spans[span_start:(span_start+span_length)] = [span]*(span_length)
        spans_id[span_start:(span_start+span_length)] = [span_start]*(span_length)
        span_length = 0   
    return spans,spans_id

def build_parse_args():
    parser = argparse.ArgumentParser(description='evaluate the model')
    parser.add_argument('--model_path', type=str, default='model_path', help='model path')
    parser.add_argument('--data_path', type=str, default='data_path', help='input data path')
    parser.add_argument('--tokenizer_type', type=str, default='local', help='tokenizer type, bert or nltk or local')
    parser.add_argument('--output_path', type=str, default='output_data', help='save the result to output_path')
    return parser.parse_args()



def main(): 
    # parse the arguments
    args = build_parse_args()
    # print the arguments
    print(args)

    id2query_path = os.path.join(args.data_path, 'id2query.json')
    labeled_file_path = os.path.join(args.data_path, 'labeledDatapoints.jsonl')

    # 1. load data from AML data labeling tool
    data = DataGenerator(id2query_path, labeled_file_path, "labeled").data
    
    raw_datasets = GetTokenTag(data, 'labeled', args.tokenizer_type, args.model_path).token_tag
    #df_labeled = pd.DataFrame(labeled)
    #df_labeled.to_csv('output_data/labeled.csv', index=False)

    
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
        span_pred, spanId_pred = get_span(token,tag_pred)
        queryId = [dct['queryId']]*len(token)
        tag_labeled = dct['tag']
        if len(tag_labeled) != len(tag_pred):
            print(f"tag_labeled length diffs, QueryId: {i}")
        
        span_labeled, spanId_labeled = get_span(token, tag_labeled)

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

    # 7. print the evaluation result

    print(classification_report(lst_label,lst_pred))

    

if __name__ == '__main__':
    main()
