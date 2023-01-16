import argparse
import os
import sys
import pandas as pd

# import self defined modules
from data_generator import DataGenerator
from get_token_tag import GetTokenTag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id2query_path', type=str, default='test_data/cjo_id2query.json', help='path to the id2query.json file')
    parser.add_argument('--predicted_old_file_path', type=str, default='test_data/cjo_result.json', help='path to the annotation result file')
    parser.add_argument('--predicted_new_file_path', type=str, default='test_data/cjo_result_newNer.json', help='path to the annotation new result file')
    parser.add_argument('--labeled_file_path', type=str, default='test_data/cjo_labeledDatapoints.jsonl', help='path to the labeled file')
    parser.add_argument('--tokenizer_type', type=str, default='nltk', help='tokenizer type, bert or nltk')
    args = parser.parse_args()
    
    data = DataGenerator(args.id2query_path, args.predicted_new_file_path, "predicted").data
    predicted_new = GetTokenTag(data, 'predicted', args.tokenizer_type).token_tag
    data = DataGenerator(args.id2query_path, args.predicted_old_file_path, "predicted").data
    predicted_old = GetTokenTag(data, 'predicted', args.tokenizer_type).token_tag
    data = DataGenerator(args.id2query_path, args.labeled_file_path, "labeled").data
    labeled = GetTokenTag(data, 'labeled', args.tokenizer_type).token_tag
    #df_labeled = pd.DataFrame(labeled)
    #df_labeled.to_csv('output_data/labeled.csv', index=False)

    assert len(predicted_new) == len(predicted_old) == len(labeled), "predicted_new, predicted_old and labeled should have same length"

    def _expand_list(lst):
        lst_new = []
        for item in lst:
            if isinstance(item, list):
                lst_new += item
            else:
                lst_new.append(item)
        return lst_new
    
    def _expand_list_single2multi(lst_multitags, lst_singletag):
        """
        Here lst_singletag is with single tag for each token, lst_multitags is with multiple tags for each token
        """
        lst_new = []
        for i, item in enumerate(lst_singletag):
            if isinstance(lst_multitags[i], list):
                l = len(lst_multitags[i])
                lst_new += [item]*l
            else:
                lst_new.append(item)
        return lst_new
    
    def _expand_predold_by_prednew(lst_pred_new, lst_pred_old):
        """
        Here lst_pred_new is based on lst_pred_old (add new tags)
        """
        lst_new = []
        for i, item in enumerate(lst_pred_old):
            if isinstance(lst_pred_new[i], list):
                l = len(item)
                if item == lst_pred_new[i]:
                    lst_new += [item]
                else:
                    lst_new += [item]*l
            else:
                lst_new.append(item)
        return lst_new
    # combine predicted_new and predicted_old and labeled
    QueryId = []
    Tokens = []
    Tags_PredNew = []
    Tags_PredOld = []
    Tags_Labeled = []
    Spans_PredNew = []
    Spans_PredOld = []
    Spans_Labeled = []
    SpansId_PredNew =[]
    SpansId_PredOld =[]
    SpansId_Labeled =[]
    for i, dct in enumerate(predicted_new):
        lst_base = dct['tag']
        token = _expand_list_single2multi(lst_base,dct['token'])
        tag_new = _expand_list(dct['tag'])
        span_new = _expand_list(dct['span'])
        spanId_new = _expand_list(dct['spanId'])
        queryId = [dct['queryId']]*len(token)
        try:
            tag_old = _expand_predold_by_prednew(lst_base,predicted_old[i]['tag'])
            span_old = _expand_predold_by_prednew(lst_base,predicted_old[i]['span'])
            spanId_old = _expand_predold_by_prednew(lst_base,predicted_old[i]['spanId'])
            tag_labeled = _expand_list_single2multi(lst_base,labeled[i]['tag'])
            span_labeled = _expand_list_single2multi(lst_base,labeled[i]['span'])
            spanId_labeled = _expand_list_single2multi(lst_base,labeled[i]['spanId'])
        except:
            print('error for queryId: ', dct['queryId'])

        try:
            Tags_Labeled += tag_labeled
        except:
            print('error for queryId: ', dct['queryId'])
            print('lst_base: ', lst_base)
            print('original labeled tag: ', labeled[i]['tag'])
        
"""
        QueryId += queryId
        Tokens += token
        Tags_PredNew += tag_new
        Tags_PredOld += tag_old
        Tags_Labeled += tag_labeled
        Spans_PredNew += span_new
        Spans_PredOld += span_old
        Spans_Labeled += span_labeled
        SpansId_PredNew += spanId_new
        SpansId_PredOld += spanId_old
        SpansId_Labeled += spanId_labeled
    assert len(Tags_PredNew) == len(Tags_PredOld) == len(Tags_Labeled), "Tags_PredNew, Tags_PredOld and Tags_Labeled should have same length"
    
    df = pd.DataFrame({'QueryId':QueryId, 'Tokens':Tokens, 'Tags_PredNew':Tags_PredNew, 'Tags_PredOld':Tags_PredOld, 
                       'Tags_Labeled':Tags_Labeled, 'Spans_PredNew':Spans_PredNew, 'Spans_PredOld':Spans_PredOld,
                        'Spans_Labeled':Spans_Labeled, 'SpansId_PredNew':SpansId_PredNew, 'SpansId_PredOld':SpansId_PredOld, 
                        'SpansId_Labeled':SpansId_Labeled})
    
    output_file = os.path.join('output_data', 'data.csv')
    df.to_csv(output_file, index=False)
    
"""
if __name__ == '__main__':
    main()
