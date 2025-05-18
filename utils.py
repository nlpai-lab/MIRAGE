"""
Utility functions for file operations, JSON/YAML handling, and argument parsing.

This module provides helper functions to:
- Check and create directories.
- Load and save JSON and JSONL files.
- Load YAML files.
- Convert string arguments to boolean.
"""

from argparse import ArgumentTypeError
import json, yaml, os, jsonlines
from typing import Any, Dict, List, Union
from datasets import Dataset

def convert_doc_pool(dataset: Dataset) -> List[Dict[str, Any]]:
    mapped_id = []
    doc_name = []
    doc_chunk = []
    support = []
    collected = []

    for data_dict in dataset['doc_pool']:
        mapped_id.extend(data_dict['mapped_id'])
        doc_name.extend(data_dict['doc_name'])
        doc_chunk.extend(data_dict['doc_chunk'])
        support.extend(data_dict['support'])

    for i in range(len(mapped_id)):
        collected.append({
            'mapped_id': mapped_id[i],
            'doc_name': doc_name[i],
            'doc_chunk': doc_chunk[i],
            'support': support[i]
        })

    return collected

def convert_oracle(dataset: Dataset) -> Dict[str, Dict[str, Any]]:
    collected = {}

    for data_dict in dataset['oracle']:
        collected[data_dict['mapped_id']] = data_dict
    return collected

def check_and_create_directory(save_path: str) -> None:
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_file(file_path: str) -> bool:
    try:
        load_json(file_path)
        print(f"The file {os.path.basename(file_path)} already exists. Therefore, we skip this process. If you want to regenerate the result, please delete or move the existing file.")
        return True
    except FileNotFoundError:
        return False
    except ValueError as e:
        print(e)  # Log invalid JSON errors for debugging
        return False

def load_json(file_path: str) -> Union[Dict[str, Any], None]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error: Invalid JSON format in file {file_path}")


def load_jsonl(file_path: str) -> Union[List[Dict[str, Any]], None]:
    if (os.path.exists(file_path)):
        with jsonlines.open(file_path, 'r') as reader:
            return [obj for obj in reader]
    else:
        print(f"Error: File not found at {file_path}")
        return None

def save_json(data: Any, save_path: str) -> None:
    check_and_create_directory(save_path)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_jsonl(data: List[Dict[str, Any]], save_path: str) -> None:
    try:
        if os.path.exists(save_path):
            existing_data = load_jsonl(save_path)
        else:
            existing_data = []
    except Exception as e:
        print(f"Error loading existing data: {e}")
        existing_data = []

    existing_data_set = {json.dumps(item, sort_keys=True) for item in existing_data}
    new_data = [item for item in data if json.dumps(item, sort_keys=True) not in existing_data_set]

    if new_data:
        check_and_create_directory(save_path)
        with jsonlines.open(save_path, mode='a') as writer:
            writer.write_all(new_data)
    else:
        print("There is no new data to save. We skip this process.")

def load_yaml(yaml_path: str) -> Union[Dict[str, Any], None]:
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {yaml_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
        
def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')