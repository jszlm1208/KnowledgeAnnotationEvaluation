import os
import sys
import json
import pandas as pd

## import self defined modules
from data_tokenizer import Tokenizer

class GetTokenTag:
    """
    This class is used to get tokenized query and corresponding tag based on the output of annotation or labeling unified format by data_generator.py
    data can be generated by data = DataGenerator(id2query_path,file_path,source).data
    tokenizer: tokenizer function
    """
    def __init__(self, data, source=['predicted', 'labeled'], tokenizer_type=['bert', 'nltk']):
        self.data = data
        self.source = source
        self.tokenizer_type = tokenizer_type
        if source == 'predicted':
            self.token_tag = self._get_token_tag_predicted(self.data, self.tokenizer_type)
        elif source == 'labeled':
            self.token_tag = self._get_token_tag_labeled(self.data, self.tokenizer_type)
        else:
            raise ValueError('Invalid source')

    def _get_token_tag_labeled(self, data, tokenizer_type):
        data_token_tag = []
        for i, dct in enumerate(data):
            queryId = dct['queryId']
            query = dct['query']
            tokenized_query = Tokenizer(query, tokenizer_type).tokenized_text
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
                    for item in Tokenizer(query[start_index:offset], tokenizer_type).tokenized_text:
                        lst_tag.append('O')
                        lst_span.append('None')
                        lst_spanId.append(-1)
                    tokens = Tokenizer(query[offset:offset_dict[offset][0]], tokenizer_type).tokenized_text
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
            
    def _get_token_tag_predicted(self, data, tokenizer_type):
        data_token_tag = []
        for i, dct in enumerate(data):
            queryId = dct['queryId']
            query = dct['query']
            tokenized_query = Tokenizer(query, tokenizer_type).tokenized_text
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
                    tokenized_span = Tokenizer(span, tokenizer_type).tokenized_text
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