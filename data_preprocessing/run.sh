id2query_path=test_data/cjo_id2query.json
file_path=test_data/cjo_result_newNer.json
source=predicted
tokenizer_type=nltk

python3 main.py \
    --id2query_path $id2query_path \
    --file_path $file_path \
    --source $source \
    --tokenizer_type $tokenizer_type 
