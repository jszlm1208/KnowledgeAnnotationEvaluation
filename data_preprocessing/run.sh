# Below are the by default arguments for the main.py
python3 main.py \
    --id2query_path "test_data/cjo_id2query.json" \
    --predicted_old_file_path "test_data/cjo_result.json" \
    --predicted_new_file_path "test_data/cjo_result_newNer.json" \
    --labeled_file_path "test_data/cjo_labeledDatapoints.jsonl" \
    --tokenizer_type "nltk" \
    --output_path "output_data"
