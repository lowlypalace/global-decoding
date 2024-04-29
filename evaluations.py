import json
import os
from evaluate import load
from datasets import Dataset
import argparse


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
        "--input_dir",
        type=str,
        default="output",
        help="Base directory containing the datasets and decoded outputs.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Directory containing locally decoded sequences.",
    )
    parser.add_argument(
        "--global_dir",
        type=str,
        required=True,
        help="Directory containing globally decoded sequences.",
    )


    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Path to the dataset file
    file_path = os.path.join("data", "webtext.test.jsonl")
    data = load_data_from_jsonl(file_path)

    # Load the reference texts
    reference_texts = [item["text"] for item in data]

    # Convert generated texts and reference texts

    # Initialize MAUVE metric
    mauve = load("mauve")

    # TODO: Do we need to truncate the lengths of the sequences to match the reference texts?

    # Load the locally decoded strings
    local_decoding_texts = load_json_file(
        os.path.join(args.input_dir, "sequences", args.local_dir, "sequences_decoded.json")
    )

    # Load the globally decoded strings
    global_decoding_texts = load_json_file(
        os.path.join(args.input_dir, "mcmc", args.global_dir, "sampled_decoded_sequences.json")
    )

    # Get the minimum length of the texts arrays and reference arrays and slice all of the arrays to that length
    min_length = min(len(reference_texts), len(local_decoding_texts), len(global_decoding_texts))
    reference_texts = reference_texts[:min_length]
    local_decoding_texts = local_decoding_texts[:min_length]
    global_decoding_texts = global_decoding_texts[:min_length]


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


if __name__ == "__main__":
    main()
