import os
import sys
import argparse
import json
import re
from azureml.core.authentication import AzureMLTokenAuthentication
from azureml.core import Workspace, Dataset, Datastore, Experiment, ScriptRunConfig

## wrong with dataset.download('image_url', '.'), image_url is exclusively of type streamInfo. Current type is FieldType.STRING

def get_auth():
    print("Attempting to get auth")
    amlToken = os.environ['AZUREML_RUN_TOKEN']
    host = os.environ['AZUREML_SERVICE_ENDPOINT']
    sid = os.environ['AZUREML_ARM_SUBSCRIPTION']
    rg = os.environ['AZUREML_ARM_RESOURCEGROUP']
    ws = os.environ['AZUREML_ARM_WORKSPACE_NAME']
    exp_name = os.environ['AZUREML_ARM_PROJECT_NAME']
    run_id = os.environ['AZUREML_RUN_ID']

    return AzureMLTokenAuthentication.create(amlToken, None, host, sid, rg, ws, exp_name, run_id)

def get_filename(input_file_path):
    # assumes the file path has .txt extension
    return input_file_path[(input_file_path.rfind('/') + 1):-4]

def get_id2query(file_path):
    id2query = {}
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".txt"):
                file_path_temp = os.path.join(root, file)
                file_id = int(file[(file.rfind('_') + 1):-4])
                with open(file_path_temp) as f:
                    id2query[file_id] = f.read()  
 
    return id2query
def get_filename(input_filepath):
    # assumes the file path has .txt extension
    return input_filepath[(input_filepath.rfind("/") + 1):-4]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription_id', type=str)
    parser.add_argument('--resource_group', type=str)
    parser.add_argument('--workspace_name', type=str)   
   # parser.add_argument('--input_dataset', type=str)
    parser.add_argument('--output_file_path', type=str)
    args = parser.parse_args()

    subscription_id = args.subscription_id
    resource_group = args.resource_group
    workspace_name = args.workspace_name
    #input_dataset = args.input_dataset

    # get the workspace
    url = os.environ["AZUREML_SERVICE_ENDPOINT"]
    print("Service URL:" + str(url))
    location = re.compile("//(.*?)\\.").search(url).group(1)
    auth = get_auth()
    workspace = Workspace(subscription_id, resource_group, workspace_name, auth=auth, _location=location, _disable_service_check=True)
    
    #dataset = Dataset.Tabular.from_json_lines_files(path=(datastore, 'Labeling/export/export/dataset/f4b3b81d-05b9-5a0d-86e2-d1878b0a1230/3da85e00-6728-4304-b546-997568dd81ca/labeledDatapoints_1.jsonl'))
    #dataset_labeled = Dataset.File.from_files(path=(datastore, 'Labeling/export/export/dataset/f4b3b81d-05b9-5a0d-86e2-d1878b0a1230/3da85e00-6728-4304-b546-997568dd81ca/labeledDatapoints_1.jsonl'))
    #mounted_path_labeled = dataset_labeled.mount()

    # get the dataset from the datastore
    #datastore = Datastore.get(workspace, "workspaceblobstore")
    #dataset_raw = Dataset.File.from_files(path=(datastore, 'Labeling/outputs/doNotDelete/tabularDataset/conversion/UX/f4b3b81d-05b9-5a0d-86e2-d1878b0a1230'))
    #mounted_path_raw = dataset_raw.mount()
    #mounted_path_raw.start()

    # get the dataset by name
    dataset = Dataset.get_by_name(workspace, name='NER_TestSet_20230224_191830')
    dataset.download('image_url', '.')
    data = dataset.to_pandas_dataframe()
    id2query = {}
    for index, row in data.iterrows():
        text_file_path = str(row["image_url"])
        input_file_name = get_filename(text_file_path)

        print("Processing: " + input_file_name)

        # read the input file path
        input_file_path = os.path.join(os.getcwd(), text_file_path)

        # read the text content
        with open(input_file_path) as f:
            id2query[index] = f.read()

    """
    temp = str(df['image_url'][0])
    path_raw = temp[1:temp.rfind('/')]
    print(path_raw)
    datastore = Datastore.get(workspace, "workspaceblobstore")
    dataset_raw = Dataset.File.from_files(path=(datastore, path_raw))
    mounted_path_raw = dataset_raw.mount()
    mounted_path_raw.start()
    
    # get the id2query
    id2query = get_id2query(mounted_path_raw)

    mounted_path_raw.stop()
    """
    # write the id2query to a file
    with open(os.path.join(args.output_file_path,'id2query.json'), 'w') as f:
        json.dump(id2query, f)

if __name__ == '__main__':
    main()

