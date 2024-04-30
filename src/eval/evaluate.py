import json
import os
from evaluate import load
import argparse

from .download_dataset import download_dataset

from utils import save_to_json


def load_data_from_jsonl(file_path):
    """Load data from a JSON Lines file."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]
    return data


def load_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate text generation quality using MAUVE metric."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Base directory to save and load the dataset.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="webtext",
        help="Name of the dataset to use as reference.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset to use as reference.",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=1000,
        help="Number of sequences to evaluate.",
    )

    return parser.parse_args()


def evaluate(args, output_subdir, local_decoding_texts, global_decoding_texts):
    # Parse command-line arguments
    args = parse_args()
    dataset_name = args.dataset_name
    split = args.split
    num_sequences = args.num_sequences
    output_subdir = args.output_subdir

    # Download the dataset
    download_dataset(subdir="data", dataset=dataset_name, splits=[split])

    # Path to the dataset file
    file_path = os.path.join(output_subdir, f"{dataset_name}.{split}.jsonl")
    data = load_data_from_jsonl(file_path)

    # Load the reference texts
    reference_texts = [item["text"] for item in data]

    # Initialize MAUVE metric
    mauve = load("mauve")

    # Trim the sequences to the desired number of sequences
    reference_texts = reference_texts[:num_sequences]
    local_decoding_texts = local_decoding_texts[:num_sequences]
    global_decoding_texts = global_decoding_texts[:num_sequences]

    # Compute MAUVE results for locally decoded strings
    mauve_results_local = mauve.compute(
        predictions=local_decoding_texts, references=reference_texts
    )
    print("MAUVE Results for Locally Decoded Strings:", mauve_results_local)

    # Compute MAUVE results for globally decoded strings
    mauve_results_global = mauve.compute(
        predictions=global_decoding_texts, references=reference_texts
    )
    print("MAUVE Results for Globally Decoded Strings:", mauve_results_global)

    # Compare scores
    print("Comparison of Local and Global MAUVE Scores:")
    print("Local:", mauve_results_local["mauve"])
    print("Global:", mauve_results_global["mauve"])

    # Save the MAUVE results to a JSON file
    save_to_json(mauve_results_local, "mauve_results_local", output_subdir)
    save_to_json(mauve_results_global, "mauve_results_global", output_subdir)
