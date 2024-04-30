import json
import os
from evaluate import load
import argparse

from download_dataset import download_dataset


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

    return parser.parse_args()


def evaluate(args, output_subdir, local_decoding_texts, global_decoding_texts):
    # Parse command-line arguments
    args = parse_args()

    # Download the dataset
    download_dataset(subdir="data", dataset=args.dataset_name, splits=[args.split])

    # Path to the dataset file
    file_path = os.path.join(args.output_dir, f"{args.dataset_name}.{args.split}.jsonl")
    data = load_data_from_jsonl(file_path)

    # Load the reference texts
    reference_texts = [item["text"] for item in data]

    # Initialize MAUVE metric
    mauve = load("mauve")

    # Get the minimum length of the texts arrays and reference arrays and slice all of the arrays to that length
    min_num_sequences = min(
        len(reference_texts), len(local_decoding_texts), len(global_decoding_texts)
    )
    reference_texts = reference_texts[:min_num_sequences]
    local_decoding_texts = local_decoding_texts[:min_num_sequences]
    global_decoding_texts = global_decoding_texts[:min_num_sequences]

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
    with open(os.path.join(output_subdir, "mauve_results.json"), "w") as file:
        json.dump(
            {
                "mauve_results_local": mauve_results_local,
                "mauve_results_global": mauve_results_global,
            },
            file,
            indent=4,
        )
