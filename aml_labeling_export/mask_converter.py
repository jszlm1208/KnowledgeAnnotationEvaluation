from PIL import Image
import os
import shutil
import json
import rle 
import numpy as np
from azureml.core import Dataset, Datastore
from azureml.data.datapath import DataPath
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

def _convert_label_to_mask(label, image_width, image_height, dest_folder, source_datastore, label_class_mappings):
    account_url = "https://" + source_datastore.account_name + ".blob.core.windows.net"
    blob_service_client = BlobServiceClient(account_url, credential = source_datastore.account_key)
    container_client = blob_service_client.get_container_client(container = source_datastore.container_name)

    label_filename = os.path.basename(label["labelFilePath"])
    with open(label_filename, "wb") as my_blob:
        print("Downloading label stored in ", label["labelFilePath"])
        download_stream = container_client.download_blob(label["labelFilePath"])
        my_blob.write(download_stream.readall())

    with open(label_filename, "r") as rle_json:
        rle_list = json.load(rle_json)["rle"]

    counts = rle_list[::2]
    values = rle_list[1::2]
    decoded = rle.decode(values, counts)[:image_height * image_width]
    combined_mask = np.array(decoded).reshape(image_height,image_width)

    for i in np.unique(combined_mask):
        binary_mask = combined_mask == i
        binary_image = Image.fromarray(binary_mask)
        binary_image = binary_image.convert('1')
        file_path = os.path.join(dest_folder, label_class_mappings[str(i)] + ".png")
        binary_image.save(file_path)

def convert_to_mask(label_data, workspace, datastore_name, source_datastore_name, target_dir_path, label_class_mappings):

    project_datastore = None
    if datastore_name:
        print("Output Datastore: ", datastore_name)
        project_datastore = Datastore.get(workspace, datastore_name)

    source_datastore = None
    if source_datastore_name:
        print("Source Datastore: ", source_datastore_name)
        source_datastore = Datastore.get(workspace, source_datastore_name)
    
    label_objects = label_data["label"]
    input_file_name =  label_data["image_name"]
    image_width = label_data["image_width"]
    image_height = label_data["image_height"]
    dest_folder = os.path.join(".", "output", input_file_name)

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for label in label_objects:
        _convert_label_to_mask(label, image_width, image_height, dest_folder, source_datastore, label_class_mappings)

    # upload local mask files and create a FileDataset
    dataset = None
    if project_datastore:
        # upload files
        dataset = Dataset.File.upload_directory(
            src_dir="./output/", target=DataPath(project_datastore, target_dir_path), show_progress=True
        )
        # delete local mask files directory
        shutil.rmtree(dest_folder)

    return dataset