import json
import os
import logging
import numpy as np

from types import SimpleNamespace

from evaluate import load

from utils import save_to_json, timer

from .download_dataset import download_dataset


def load_data_from_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]
    return data


def load_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def convert_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: convert_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to list
    return obj


def evaluate(args, output_subdir, local_decoding_texts, global_decoding_texts):
    # Parse command-line arguments
    eval_dataset_name = args.eval_dataset_name
    eval_split = args.eval_split
    eval_num_sequences = args.eval_num_sequences

    # Set the output subdirectory
    subdir = "data"

    # Download the dataset
    logging.info(f"Downloading the {eval_dataset_name} dataset...")
    download_dataset(subdir="data", dataset=eval_dataset_name, splits=[eval_split])

    # Path to the dataset file
    file_path = os.path.join(subdir, f"{eval_dataset_name}.{eval_split}.jsonl")
    data = load_data_from_jsonl(file_path)

    # Load the reference texts
    reference_texts = [item["text"] for item in data]

    # Trim the sequences to the specified number of sequences
    reference_texts = reference_texts[:eval_num_sequences]
    local_decoding_texts = local_decoding_texts[:eval_num_sequences]
    global_decoding_texts = global_decoding_texts[:eval_num_sequences]

    with timer("Evaluating the generated sequences..."):
        # Initialize MAUVE metric
        mauve = load("mauve")
        # Compute MAUVE results for locally decoded strings
        mauve_results_local = mauve.compute(
            predictions=local_decoding_texts, references=reference_texts
        )
        # Compute MAUVE results for globally decoded strings
        mauve_results_global = mauve.compute(
            predictions=global_decoding_texts, references=reference_texts
        )

    logging.info(
        f"MAUVE score for locally decoded strings: {mauve_results_local.mauve}"
    )
    logging.info(
        f"MAUVE score for globally decoded strings: {mauve_results_global.mauve}"
    )

    # Save the MAUVE results to a JSON file
    logging.info("Saving the evaluation results...")
    mauve_results_local_dict = convert_to_dict(mauve_results_local)
    mauve_results_global_dict = convert_to_dict(mauve_results_global)
    save_to_json(mauve_results_local_dict, "mauve_results_local", output_subdir)
    save_to_json(mauve_results_global_dict, "mauve_results_global", output_subdir)

    return mauve_results_local, mauve_results_global
