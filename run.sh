python3 model_evaluate/evaluate_aml.py \
    --model_path '/share/models/deberta-xsmall-30k' \
    --id2query_path 'model_evaluate/data/cjo_id2query.json' \
    --labeled_file_path 'model_evaluate/data/cjo_labeledDatapoints.jsonl' \
    --tokenizer_type 'local' \
    --output_path 'model_evaluate/output_data'