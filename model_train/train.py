import argparse
# define a function to parse the arguments
def build_parse_args():
    parser = argparse.ArgumentParser(description='Train a model')

    ## Required parameters
    parser.add_argument("--max_epochs", default=150, type=int, required=True, help="Number of epochs to run")
    parser.add_argument("--batch_size", default=32, type=int, required=True, help="Batch size")
    parser.add_argument("--hidden_size", default=256, type=int, required=True, help="Hidden dims")
    parser.add_argument("--num_layers", default=2, type=int, required=True, help="Number of layers")
    parser.add_argument("--lr", default=0.001, type=float, required=True, help="Learning rate")

    parser.add_argument("--transformer_embedding_type", default="xlm-roberta-base", type=str,required=True,
            help="Choose type of transformer embeddings. (BERT, XLM, GPT, RoBERTa, XLNet, DistilBERT etc.)")
    
    parser.add_argument("--model_name", default="model", type=str, required=True, help="Model name")
    parser.add_argument("--data_dir", default="data", type=str, required=True, help="Data directory")
    parser.add_argument("--model_dir", default="model", type=str, required=True, help="Model directory")
    

    parser.add_argument("--train_file", type=str, required=True, help="Train file")
    parser.add_argument("--test_file", type=str, required=True, help="Test file")
    parser.add_argument("--dev_file", type=str, required=True, help="Dev file")
    
    parser.add_argument("--resource_name", default="NA", type=str, required=True,
            help="Add a resource name for your experiment. E.g. 'conll-knowledge-ner' to describe NER model on CONLL dataset with external knowledge")
    
    ## Optional parameters
    parser.add_argument("--use_crf", default="True", type=str, required=False, help="Use CRF")
    parser.add_argument("--output_dir", default="./", type=str, required=False, help="Output directory")
    return parser.parse_args()

# main function
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings, TransformerWordEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import torch, flair
from torch.optim.lr_scheduler import OneCycleLR

def main():
    # parse the arguments
    args = build_parse_args()
    # print the arguments
    print(args)

    # 1. get the corpus
    columns = {0: 'text', 3: 'ner'}
    data_folder = args.data_dir
    corpus: Corpus = ColumnCorpus(data_folder, columns, train_file=args.train_file, test_file=args.test_file, dev_file=args.dev_file)

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embeddings = TransformerWordEmbeddings(
        model=args.transformer_embedding_type,
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=False,
        respect_document_boundaries=False,
        )
    
    # 5. initialize sequence tagger
    tagger : SequenceTagger = SequenceTagger(hidden_size=args.hidden_size,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=args.use_crf,
                                        use_rnn=False,
                                        reproject_embeddings=False)
    
    # 6. initialize trainer
    trainer : ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training (fine-tuning)
    resource_path = args.output_dir + "/" + 'resources/taggers/' + args.resource_name
    trainer.fine_tune(resource_path,
                learning_rate=args.lr,
                mini_batch_size=args.batch_size,
                mini_batch_chunk_size=1,
                max_epochs=args.max_epochs,
                scheduler=OneCycleLR,
                embeddings_storage_mode='none',
                use_final_model_for_eval = True,
                weight_decay=0.)


if __name__ == '__main__':
    main()

