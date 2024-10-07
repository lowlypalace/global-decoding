import os
import json
import argparse

from src.utils.utils import get_unique_name


def find_sequences_and_probs(input_dir, top_k, top_p, model_name):
    sequences_ids = []
    sequences_decoded = []
    logprobs_target = []
    logprobs_proposal = []
    proposal_normalize_constans = []
    target_normalize_constants = []
    logprobs_proposal_tokens = []
    logprobs_target_tokens = []
    # proposal_normalize_constants_products = []
    # target_normalize_constants_products = []

    input_dir = os.path.join(input_dir, model_name)

    # List all directories in the input directory
    directories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    print(f"Top-k: {top_k}, Top-p: {top_p}")

    for directory in directories:
        metadata_file = os.path.join(input_dir, directory, "metadata.json")

        # Read metadata.json to check if the directory matches the criteria
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        if metadata["top_k"] == top_k and metadata["top_p"] == top_p and metadata["model_name"] == model_name:
            # For debugging
            print(f"Found matching directory: {directory}")

            with open(
                os.path.join(input_dir, directory, "sequences", "sequences_ids.json"),
                "r",
            ) as f:
                sequences_ids.extend(json.load(f))
            with open(
                os.path.join(input_dir, directory, "sequences", "sequences_decoded.json"),
                "r",
            ) as f:
                sequences_decoded.extend(json.load(f))
            with open(
                os.path.join(input_dir, directory, "sequences", "logprobs_target.json"),
                "r",
            ) as f:
                logprobs_target.extend(json.load(f))
            with open(
                os.path.join(input_dir, directory, "sequences", "logprobs_proposal.json"),
                "r",
            ) as f:
                logprobs_proposal.extend(json.load(f))
            with open(
                os.path.join(
                    input_dir,
                    directory,
                    "sequences",
                    "proposal_normalize_constants.json",
                ),
                "r",
            ) as f:
                proposal_normalize_constans.extend(json.load(f))
            with open(
                os.path.join(input_dir, directory, "sequences", "target_normalize_constants.json"),
                "r",
            ) as f:
                target_normalize_constants.extend(json.load(f))

            with open(
                os.path.join(input_dir, directory, "sequences", "logprobs_proposal_tokens.json"),
                "r",
            ) as f:
                logprobs_proposal_tokens.extend(json.load(f))
            with open(
                os.path.join(input_dir, directory, "sequences", "logprobs_target_tokens.json"),
                "r",
            ) as f:
                logprobs_target_tokens.extend(json.load(f))

            metadata_to_save = metadata

            # with open(
            #     os.path.join(input_dir, directory, "sequences", "proposal_normalize_constants_products.json"),
            #     "r",
            # ) as f:
            #     proposal_normalize_constants_products.extend(json.load(f))
            # with open(
            #     os.path.join(input_dir, directory, "sequences", "target_normalize_constants_products.json"),
            #     "r",
            # ) as f:
            #     target_normalize_constants_products.extend(json.load(f))

    # For debugging
    print(f"Found {len(sequences_ids)} sequences ids")
    print(f"Found {len(sequences_decoded)} decoded sequences")
    print(f"Found {len(logprobs_target)} target logprobs")
    print(f"Found {len(logprobs_proposal)} proposal logprobs")
    print(f"Found {len(proposal_normalize_constans)} proposal normalize constants")
    print(f"Found {len(target_normalize_constants)} target normalize constants")

    return (
        sequences_ids,
        sequences_decoded,
        logprobs_proposal,
        logprobs_target,
        proposal_normalize_constans,
        target_normalize_constants,
        logprobs_proposal_tokens,
        logprobs_target_tokens,
        # proposal_normalize_constants_products,
        # target_normalize_constants_products,
        metadata_to_save,
    )


def save_merged_sequences(
    input_dir,
    model_name,
    sequences_ids,
    sequences_decoded,
    logprobs_proposal,
    logprobs_target,
    proposal_normalize_constants,
    target_normalize_constants,
    logprobs_proposal_tokens,
    logprobs_target_tokens,
    # proposal_normalize_constants_products,
    # target_normalize_constants_products,
    metadata,
):
    # Create the output directory
    output_dir = os.path.join(input_dir, "merged", model_name, get_unique_name())
    os.makedirs(output_dir, exist_ok=True)

    sequences_dir = os.path.join(output_dir, "sequences")
    os.makedirs(sequences_dir, exist_ok=True)

    # Save sequences and probabilities
    with open(os.path.join(sequences_dir, "sequences_ids.json"), "w") as f:
        json.dump(sequences_ids, f)

    with open(os.path.join(sequences_dir, "sequences_decoded.json"), "w") as f:
        json.dump(sequences_decoded, f)

    with open(os.path.join(sequences_dir, "logprobs_target.json"), "w") as f:
        json.dump(logprobs_target, f)

    with open(os.path.join(sequences_dir, "logprobs_proposal.json"), "w") as f:
        json.dump(logprobs_proposal, f)

    with open(os.path.join(sequences_dir, "proposal_normalize_constants.json"), "w") as f:
        json.dump(proposal_normalize_constants, f)

    with open(os.path.join(sequences_dir, "target_normalize_constants.json"), "w") as f:
        json.dump(target_normalize_constants, f)

    with open(os.path.join(sequences_dir, "logprobs_proposal_tokens.json"), "w") as f:
        json.dump(logprobs_proposal_tokens, f)

    with open(os.path.join(sequences_dir, "logprobs_target_tokens.json"), "w") as f:
        json.dump(logprobs_target_tokens, f)

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved merged sequences and metadata to {output_dir}")


def main():
    # Example usage with args
    parser = argparse.ArgumentParser(description="Merging sequences")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="output",
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pythia-70m",
        choices=[
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "pythia-70m",
            "pythia-160m",
            "pythia-410m",
            "pythia-1b",
            "pythia-1.4b",
            "pythia-2.8b",
            "pythia-6.9b",
            "pythia-12b",
        ],
        help="Model to use for text generation. Supports GPT-2 and Pythia.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k value for text generation. No default value; must be specified if used.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p value for text generation. No default value; must be specified if used.",
    )

    args = parser.parse_args()

    # Find and merge sequences
    (
        sequences_ids,
        sequences_decoded,
        logprobs_proposal,
        logprobs_target,
        proposal_normalize_constants,
        target_normalize_constants,
        logprobs_proposal_tokens,
        logprobs_target_tokens,
        # proposal_normalize_constants_products,
        # target_normalize_constants_products,
        metadata,
    ) = find_sequences_and_probs(args.input_dir, args.top_k, args.top_p, args.model_name)

    # Save the merged sequences and metadata
    save_merged_sequences(
        args.input_dir,
        args.model_name,
        sequences_ids,
        sequences_decoded,
        logprobs_proposal,
        logprobs_target,
        proposal_normalize_constants,
        target_normalize_constants,
        logprobs_proposal_tokens,
        logprobs_target_tokens,
        # proposal_normalize_constants_products,
        # target_normalize_constants_products,
        metadata,
    )


if __name__ == "__main__":
    main()
