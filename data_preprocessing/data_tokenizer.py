from transformers import AutoTokenizer
# use word_tokenize to tokenize the text
from nltk.tokenize import word_tokenize
import re

class Tokenizer:
    # lst_char = ['.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '/', '\\', '"', "'", '-', '_', '+', '=', '*', '&', '^', '%', '$', '#', '@', '~', '`', '<', '>', '|', ' ']
    def __init__(self, text, tokenizer_type=['bert', 'nltk','local'], model_path=None):
        lst_char = ['"', "'"]
        pattern = '[' + ''.join(lst_char) + ']'
        self.tokenizer_type = tokenizer_type
        self.text = text
        tokenized_text = self.tokenize_text(self.text, self.tokenizer_type)
        self.tokenized_text = [re.sub(pattern, '', token) for token in tokenized_text]
        self.model_path = model_path

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


