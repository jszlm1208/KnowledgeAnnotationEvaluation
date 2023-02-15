python3 evaluate_aml.py \
    --model_path '/home/bincao/code/transformers/examples/pytorch/token-classification/deberta-xsmall-ontonotes-run1' \
    --id2query_path 'data/cjo_id2query.json' \
    --labeled_file_path 'data/cjo_labeledDatapoints.jsonl' \
    --tokenizer_type 'local' \
    --output_path 'output_data'