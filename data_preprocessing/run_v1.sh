# Below are the by default arguments for the main.py
SEGMENT="af"
python3 main_v1.py \
    --id2query_path "test_data/${SEGMENT}_id2query.json" \
    --predicted_old_file_path "test_data/${SEGMENT}_result.json" \
    --predicted_new_file_path "test_data/${SEGMENT}_result_newNer.json" \
    --labeled_file_path "test_data/${SEGMENT}_labeledDatapoints.jsonl" \
    --tokenizer_type "nltk" \
    --output_path "output_data"
