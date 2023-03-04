import os
import sys
import argparse
import json
import re
from azureml.core.authentication import AzureMLTokenAuthentication
from azureml.core import Workspace, Dataset

from conll_converter import convert_to_conll
from tabular_converter import convert_to_csv

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

def export_to_conll(dataset, output_file_path):    
    data = dataset.to_pandas_dataframe()
    print("Number of datapoints to process: " + str(len(data.index)))
    dataset.download('image_url', '.')
    output_file_path = os.path.join(os.getcwd(), output_file_path)

    for index, row in data.iterrows():
        print("Processing file #" + str(index))
        label_objects = row["label"]
        input_file_path = str(row["image_url"])
        convert_to_conll(input_file_path, label_objects, output_file_path)

def export_to_csv(dataset, output_file_path, project_type):
    data = dataset.to_pandas_dataframe()
    print("Number of datapoints to process: " + str(len(data.index)))
    dataset.download('image_url', '.')
    column_names = data.columns.to_list()
    output_file_path = os.path.join(os.getcwd(), output_file_path)

    for index, row in data.iterrows():
        print("Processing file #" + str(index))
        # skip the 1st column_names, which is image_url
        convert_to_csv(row, column_names[1:], output_file_path, index == 0, project_type)

def main():
    print("Script execution started")
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription_id', type=str)
    parser.add_argument('--resource_group', type=str)
    parser.add_argument('--workspace_name', type=str)
    parser.add_argument('--project_type', type=str)
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--input_dataset', type=str)
    parser.add_argument('--export_format', type=str)
    parser.add_argument('--output_file_path', type=str)             # CoNLL
    parser.add_argument('--datastore_name', type=str)               # Bit Mask
    parser.add_argument('--target_dir_path', type=str)              # Bit Mask
    parser.add_argument('--output_dataset_suffix', type=str)        # Bit Mask
    parser.add_argument('--output_dataset_tags', type=json.loads)   # Bit Mask
    parser.add_argument('--label_class_mappings', type=json.loads)  # Bit Mask

    args=parser.parse_args()

    subscription_id = args.subscription_id
    resource_group = args.resource_group
    workspace_name = args.workspace_name
    project_type = args.project_type
    project_name = args.project_name
    dataset_name = args.input_dataset

    url = os.environ["AZUREML_SERVICE_ENDPOINT"]
    print("Service URL:" + str(url))
    location = re.compile("//(.*?)\\.").search(url).group(1)
    auth = get_auth()

    workspace = Workspace(subscription_id, resource_group, workspace_name, auth=auth, _location=location, _disable_service_check=True)
    dataset = Dataset.get_by_name(workspace, dataset_name)
    print("Fetched the dataset: " + dataset_name)
    print("Project type: " + project_type)

    if (project_type == "TextNamedEntityRecognition"):
        if (args.export_format == "CoNLL"):
            export_to_conll(dataset, args.output_file_path)
        else:
            export_to_csv(dataset, args.output_file_path, project_type)

    elif project_type == "SemanticSegmentation":

        from mask_converter import convert_to_mask

        data = dataset._dataflow.to_pandas_dataframe(extended_types=True)
        print("Number of datapoints to process: " + str(len(data.index)))

        datastore_name = args.datastore_name
        target_dir_path = args.target_dir_path
        output_dataset_suffix = args.output_dataset_suffix
        output_dataset_tags = args.output_dataset_tags
        label_class_mappings = args.label_class_mappings
        source_datastore = output_dataset_tags["SourceDatastoreName"]

        # for all label datapoint enteries, convert them to bit masks
        # note: if datastore_name is not None, generated bit-masks are uploaded to the
        # given datastore, and a file dataset of bit-masks is created and returned
        export_dataset = None
        for index, row in data.iterrows():
            print("Processing entry #" + str(index))
            export_dataset = convert_to_mask(row, workspace, datastore_name, source_datastore, target_dir_path, label_class_mappings)

        # if we have an output dataset, register it to the workspace
        if export_dataset:

            if output_dataset_suffix == None:
                output_dataset_suffix = "_masks"

            export_dataset_name = dataset_name + output_dataset_suffix
            description = "Exported bit masks for dataset: " + dataset_name
            if project_name:
                description = description + ", project: " + project_name

            export_dataset.register(
                workspace=workspace,
                name=export_dataset_name,
                description=description,
                tags=output_dataset_tags
            )
    elif (project_type == "TextClassificationMultiClass") or (project_type == "TextClassificationMultiLabel"):
        export_to_csv(dataset, args.output_file_path, project_type)
    else:
        print("Conversion for project type " + project_type + " is not supportd or not implemented yet!")

    print("Script execution completed.")


if __name__ == "__main__":
    main()