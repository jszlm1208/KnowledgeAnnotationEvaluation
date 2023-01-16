python3 main.py \
    --id2query_path "test_data/cjo_id2query.json" \
    --file_path "test_data/cjo_result.json" \
    --source "predicted" \
    --tokenizer_type "nltk"

python3 main.py \
    --id2query_path "test_data/cjo_id2query.json" \
    --file_path "test_data/cjo_result_newNer.json" \
    --source "predicted" \
    --tokenizer_type "nltk"

# Path: run.sh
python3 main.py \
    --id2query_path "test_data/cjo_id2query.json" \
    --file_path "test_data/cjo_labeledDatapoints.jsonl" \
    --source "labeled" \
    --tokenizer_type "nltk" \

