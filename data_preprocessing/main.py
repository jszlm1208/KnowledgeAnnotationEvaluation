import argparse
import os
import sys
import pandas as pd

# import self defined modules
from data_generator import DataGenerator
from get_token_tag import GetTokenTag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id2query_path', type=str, default='data/id2query.json', help='path to the id2query.json file')
    parser.add_argument('--file_path', type=str, default='data/annotation.json', help='path to the annotation.json file')
    parser.add_argument('--source', type=str, default='predicted', help='source of the data, predicted or labeled')
    parser.add_argument('--tokenizer_type', type=str, default='nltk', help='tokenizer type, bert or nltk')
    args = parser.parse_args()

    data = DataGenerator(args.id2query_path, args.file_path, args.source).data
    token_tag = GetTokenTag(data, args.source, args.tokenizer_type).token_tag
    df = pd.DataFrame(token_tag)
    df.to_csv('data/token_tag.csv', index=False)

if __name__ == '__main__':
    main()
