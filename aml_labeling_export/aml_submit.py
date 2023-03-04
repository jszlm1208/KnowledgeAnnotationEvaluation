import argparse

from azureml.core import Workspace, Dataset, Datastore, Experiment, ScriptRunConfig
import json

def build_parse_args():
    parser = argparse.ArgumentParser(description='export labeling data')
    parser.add_argument('--subscription_id', type=str, default='')
    parser.add_argument('--resource_group', type=str, default='')
    parser.add_argument('--workspace_name', type=str, default='')
    parser.add_argument('--project_type', type=str, default='TextNamedEntityRecognition')
    parser.add_argument('--project_name', type=str, default='GoldSet_NER')
    parser.add_argument('--input_dataset', type=str, default='GoldSet_NER_20230222_185138')
    parser.add_argument('--export_format', type=str, default='csv')
    parser.add_argument('--output_file_path', type=str, default='outputs')
    return parser.parse_args()

def main():
    args = build_parse_args()
    print(args)
    
    subscription_id = args.subscription_id
    resource_group = args.resource_group
    workspace_name = args.workspace_name
    workspace = Workspace(subscription_id, resource_group, workspace_name)
    experiment = Experiment(workspace, "aml_labeling_export")
    
    # run experiments
    arguments = [   "--subscription_id", args.subscription_id,
                    "--resource_group", args.resource_group,
                    "--workspace_name", args.workspace_name,
                    "--project_type", args.project_type,
                    "--project_name", args.project_name,
                    "--input_dataset", args.input_dataset,
                    "--export_format", args.export_format,
                    "--output_file_path", args.output_file_path   
                ]
    config = ScriptRunConfig(source_directory="./",
                             script="export_converter.py",
                             arguments=arguments,
                             compute_target="BizQADevWUS3CPU")
    env = workspace.environments['LabelingExport']
    config.run_config.environment = env
    run = experiment.submit(config)

    # get the url of the run
    aml_url = run.get_portal_url()
    print(aml_url)
    # run.wait_for_completion(show_output=True)

if __name__ == "__main__":
    main()