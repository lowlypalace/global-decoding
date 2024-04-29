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
        default="sequences",
        help="Subdirectory name for locally decoded sequences.",
    )
    parser.add_argument(
        "--global_dir",
        type=str,
        default="mcmc",
        help="Subdirectory name for globally decoded sequences.",
    )
    parser.add_argument(
        "--ref_file",
        type=str,
        default="webtext.test.jsonl",
        help="Filename for the reference strings data.",
    )
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Path to the dataset file
    file_path = os.path.join("data", "webtext.test.jsonl")
    data = load_data_from_jsonl(file_path)

    # Let's assume each line in JSON has a key 'text' that contains the textual data.
    texts = [item["text"] for item in data]
    dataset = Dataset.from_dict({"text": texts})

    # Convert generated texts and reference texts
    generated_texts = [text.strip() for text in generated_texts]
    reference_texts = [text["text"].strip() for text in dataset[:10]["text"]]

    # Initialize MAUVE metric
    mauve = load("mauve")

    # Load the locally decoded strings
    local_decoding_texts = load_json_file(
        os.path.join(args.input_dir, args.local_dir, "sequences_decoded.json")
    )

    # Load the globally decoded strings
    global_decoding_texts = load_json_file(
        os.path.join(args.input_dir, args.global_dir, "sampled_decoded_sequences.json")
    )

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
