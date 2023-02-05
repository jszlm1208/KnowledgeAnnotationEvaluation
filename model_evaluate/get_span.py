import re
from pandas import json_normalize
import json
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

if __name__ == '__main__':
    """
    #write test cases to file
    tokens = ['▁mi', 'ke', '▁is', '▁assistant', '▁to', '▁which', '▁listed', '▁customer', '▁who', '▁graduated', '▁mit', '?', ]
    #['▁contact', 's', '▁in', '▁Paris', '▁France']
    labels = ['B-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ]
    #['O', 'O', 'O', 'B-GPE', 'B-GPE']
    spans,spans_id = get_span(tokens,labels)
    output = {'tokens':tokens,'labels':labels,'spans':spans}
    output = json.dumps(output)
    with open('get_span_testcases.json','a') as f:
        f.write(output)
        f.write('\n')
    """
    #read test cases from file to check
    with open('get_span_testcases.json','r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            tokens = line['tokens']
            labels = line['labels']
            spans = line['spans']
            spans_new,_ = get_span(tokens,labels)
            if spans_new != spans:
                print('tokens: ',tokens)
                print('labels: ',labels)
                print('spans: ',spans)
                print('spans_new: ',spans_new)
                print('-----------------')
            else:
                print('pass')
        
    
