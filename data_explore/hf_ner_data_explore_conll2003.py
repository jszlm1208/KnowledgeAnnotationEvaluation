## data explore for huggingface NER dataset
from datasets import load_dataset
import pandas as pd
import argparse
import sys
import os
import json

# import self defined modules/functions
sys.path.append('../utils')
from data_utils import get_span, tokens_to_query

def build_parse_args():
    parser = argparse.ArgumentParser(description='data explore for huggingface NER dataset')
    parser.add_argument('--dataset_name', type=str, default='conll2003', help='dataset name')
    parser.add_argument('--output_path', type=str, default='output_data', help='save the result to output_path')
    return parser.parse_args()

def main():
    # parse the arguments
    args = build_parse_args()
    # print the arguments
    print(args)
    
    # load dataset
    dataset = load_dataset(args.dataset_name)

    
    # get the train, test, validation data
    train = dataset['train']
    test = dataset['test']
    validation = dataset['validation']

    # get id2label and label2id
    id2label = dict(zip(list(range(train.features['ner_tags'].feature.num_classes)),train.features['ner_tags'].feature.names))
    label2id = dict(zip(train.features['ner_tags'].feature.names,list(range(train.features['ner_tags'].feature.num_classes))))
    
    # convert to expanded dataframe
    def _expand_df(data):
        QueryId = []
        Tokens = []
        Tags_Labeled = []
        Spans_Labeled = []
        SpansId_Labeled = []
        id2query = {}
        for i, dct in enumerate(data):
            token = dct['tokens']
            tag_labeled = [*map(id2label.get,dct['ner_tags'])]
            span_labeled, spanId_labeled = get_span(token, tag_labeled)
            queryId = [dct['id']]*len(token)
            
            QueryId += queryId
            Tokens += token
            Tags_Labeled += tag_labeled
            Spans_Labeled += span_labeled
            SpansId_Labeled += spanId_labeled

            id2query[i] = tokens_to_query(token)
        df = pd.DataFrame({'QueryId':QueryId, 'Tokens':Tokens, 'Tags_Labeled':Tags_Labeled, 'Spans_Labeled':Spans_Labeled, 'SpansId_Labeled':SpansId_Labeled})
        df['OrderId'] = df.index
        return df, id2query

    df_train, id2query_train = _expand_df(train)
    df_test, id2query_test = _expand_df(test)
    df_validation, id2query_validation = _expand_df(validation)

   
    # check if output_path exists, if not, create it
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
   
    
    # save to csv
    df_train.to_csv(os.path.join(args.output_path,'df_train.csv'), index=False)
    df_test.to_csv(os.path.join(args.output_path,'df_test.csv'), index=False)
    df_validation.to_csv(os.path.join(args.output_path,'df_validation.csv'), index=False)

    # save id2query to json
    with open(os.path.join(args.output_path,'id2query_train.json'), 'w') as f:
        json.dump(id2query_train, f)
    
    with open(os.path.join(args.output_path,'id2query_test.json'), 'w') as f:
        json.dump(id2query_test, f)
    
    with open(os.path.join(args.output_path,'id2query_validation.json'), 'w') as f:
        json.dump(id2query_validation, f)

if __name__ == '__main__':
    main()