import re
from pandas import json_normalize
import json
import os


# used for file type data labeling project, to get the id2query
def get_id2query(file_path):
    id2query = {}
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".txt"):
                file_path_temp = os.path.join(root, file)
                file_id = int(file[(file.rfind('_') + 1):-4])
                with open(file_path_temp) as f:
                    id2query[file_id] = f.read()  
 
    return id2query

# read csv row and write to txt file. Used for labeling
def csv_to_txt(file_path, file):
    """
    file_path: path to the folder that contains the csv file
    file: csv file name like 'data.csv'
    """
    data = pd.read_csv(os.path.join(file_path, file))
    for index, row in data.iterrows():
        txt_file = file[:-4] + '_' + str(index) + '.txt'
        output_path = os.path.join(file_path, file[:-4])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(os.path.join(output_path,txt_file), 'w') as f:
            f.write(row[0])

def tokens_to_query(tokens):
    query = ""
    lst_special = ['.', ',', '!', '?', ';',  '[', ']', '{', '}'
                   , '\\', "'", '_', '+', '*', '&', '^', '%', '...'
                   , '#', '@', '~', '`', '<', '>', '|', ' ', "'s", "n't"]
    for token in tokens:
        if token in lst_special or query=="":
            query += token
        else:
            query += ' ' + token  
    return query

def get_span(tokens,labels,tokens_type='word'):
    #tokens_type: word, subword. If subword, then need to do re.sub("▁"," ",span)
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
                if tokens_type == 'subword':
                    span = ''.join(tokens[span_start:(span_start+span_length)])
                    span = re.sub("▁"," ",span)
                    span = span.strip()
                elif tokens_type == 'word':
                    span = ' '.join(tokens[span_start:(span_start+span_length)])
                    span = span.strip()
                else:
                    raise ValueError("tokens_type must be either word or subword")
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
                if tokens_type == 'subword':
                    span = ''.join(tokens[span_start:(span_start+span_length)])
                    span = re.sub("▁"," ",span)
                    span = span.strip()
                elif tokens_type == 'word':
                    span = ' '.join(tokens[span_start:(span_start+span_length)])
                    span = span.strip()
                else:
                    raise ValueError("tokens_type must be either word or subword")
                spans[span_start:(span_start+span_length)] = [span]*(span_length)
                spans_id[span_start:(span_start+span_length)] = [span_start]*(span_length)
                span_length = 0                   
    #check the end tag
    if span_length > 0:
        if tokens_type == 'subword':
                    span = ''.join(tokens[span_start:(span_start+span_length)])
                    span = re.sub("▁"," ",span)
                    span = span.strip()
        elif tokens_type == 'word':
            span = ' '.join(tokens[span_start:(span_start+span_length)])
            span = span.strip()
        else:
            raise ValueError("tokens_type must be either word or subword")
        spans[span_start:(span_start+span_length)] = [span]*(span_length)
        spans_id[span_start:(span_start+span_length)] = [span_start]*(span_length)
        span_length = 0   
    return spans,spans_id


if __name__ == '__main__':
    # define tokens_type
    tokens_type = 'word'
    if tokens_type == 'subword':
        testcases = 'get_span_subword_testcases.json'
    elif tokens_type == 'word':
        testcases = 'get_span_word_testcases.json'
    else:
        raise ValueError("tokens_type must be either word or subword")
    
    """
    #----------------------------------------------------------------------------------------------------
    #write test cases to file for get_span function
    tokens = ['Tokens', 'JOSE', 'Eduardo', 'Dutra', ',', 'who', 'had', 'drawn', 'up', 'the', 'Bill', '.']
    #['▁contact', 's', '▁in', '▁Paris', '▁France']
    labels = ['B-PERSON', 'I-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    #['O', 'O', 'O', 'B-GPE', 'B-GPE']
    spans,spans_id = get_span(tokens,labels, tokens_type)
    output = {'tokens':tokens,'labels':labels,'spans':spans}
    print(output)

    output = json.dumps(output)
    with open(testcases,'a') as f:
        f.write(output)
        f.write('\n')
    
    """
    #----------------------------------------------------------------------------------------------------
    #read test cases from file to check
    with open(testcases,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            tokens = line['tokens']
            labels = line['labels']
            spans = line['spans']
            spans_new,_ = get_span(tokens,labels, tokens_type)
            if spans_new != spans:
                print('tokens: ',tokens)
                print('labels: ',labels)
                print('spans: ',spans)
                print('spans_new: ',spans_new)
                print('-----------------')
            else:
                print('pass')
    
        
    
