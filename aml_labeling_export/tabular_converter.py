import os
import json
import csv
import numpy as np

def get_filename(input_filepath):
    # assumes the file path has .txt extension
    return input_filepath[(input_filepath.rfind("/") + 1):-4]

def convert_to_csv(label_data, column_names, output_file_path, is_first_row, project_type) :

    text_file_path = str(label_data["image_url"])
    input_file_name = get_filename(text_file_path)

    print("Processing: " + input_file_name)

    # read the input file path
    input_file_path = os.path.join(os.getcwd(), text_file_path)

    # read the text content
    with open(input_file_path) as f:
        input_text_content = f.read()
    
    header = []
    row = []
    data = []
    # The raw data is saved to each file per row for text NER project
    # The tabular dataset row data is saved to each file per row for text classification project
    if project_type == "TextNamedEntityRecognition":
        data = input_text_content
    else:
        data = json.loads(input_text_content)
    
    if is_first_row:
        if project_type == "TextNamedEntityRecognition":
            header = ["data"]
        else:
            header = list(data["columnOrdinals"].keys())
        header.extend(column_names)
    
    print (header)
    if project_type == "TextNamedEntityRecognition":
        row.append(data)
    else:
        for column in data["columnValues"]:
            # The value might not exist because json ingores the null value
            if "value" in column:
                row.append(column["value"])
            else:
                row.append("")
    
    for column_name in column_names:
        if type(label_data[column_name]) == np.ndarray:
            data = ",".join([str(value) for value in label_data[column_name]])
            row.append(data)
        else:
            row.append(label_data[column_name])

    print (row)
    with open(output_file_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)

        # write the header
        if (is_first_row):
            writer.writerow(header)

        # write the data
        writer.writerow(row)