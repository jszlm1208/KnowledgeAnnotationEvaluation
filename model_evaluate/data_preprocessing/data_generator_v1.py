import json
import os
import sys
import pandas as pd
import numpy as np
import re


class DataGenerator:
    """"
    This class is used to convert data output by annotation and by labeling to unified format.
    id2query: file that maps query id to query, json file
    file_path: file that contains data output by annotation or by labeling, json file
    source: ['predicted', 'labeled'], if 'predicted', then file_path is the output of annotation, if 'labeled', then file_path is the output of labeling
    """
    def __init__(self, id2query,file_path, source=['predicted', 'labeled']):
        self.id2query = id2query
       
        if source == 'predicted':
            self.data, self.lst_error = self._read_predictedfile(file_path)
        elif source == 'labeled':
            self.data, self.lst_error = self._read_labeledfile(file_path)
        else:
            raise ValueError('source should be either predicted or labeled')

        assert len(self.data) == len(self.id2query), "output file and id2query file should have same length"

    
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
                    data[int(temp['queryId'])]= temp # order the labeled results by queryId
                except:
                    lst_error.append(json.loads(line))
        return data, lst_error
