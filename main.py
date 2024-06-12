import torch
import argparse
import logging
import secrets
import os


from src.utils.utils import (
    setup_logging,
    save_args,
    set_seed,
    load_from_json,
)
from src.utils.validate import validate_args

from src.sequences import generate_sequences_and_probs
from src.mcmc import run_mcmc
from src.eval import evaluate


# Define the function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text sequences, run MCMC, and evaluate the results."
    )

    # Sequence generation arguments
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
        "--text",
        type=str,
        default=None,
        help="Text to use as a prompt. Defaults to the EOS token.",
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
    parser.add_argument(
        "--sequence_count",
        type=int,
        default=1000,
        help="Number of sequence samples to generate and use for MCMC analysis.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length. If not provided, it will be set to the maximum model length minus the length of the input text.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to use for computation. Defaults to "cuda" if available.',
    )
    parser.add_argument(
        "--batch_size_seq",
        type=int,
        default=64,
        help="Batch size for generating sequences.",
    )
    parser.add_argument(
        "--batch_size_prob",
        type=int,
        default=16,
        help="Batch size for computing probabilities.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp64",
        choices=["fp16", "fp32", "fp64"],
        help="Precision to use for the model. Defaults to fp64.",
    )

    parser.add_argument(
        "--preload_dir",
        type=str,
        default=None,
        help="Directory name to preload generated sequences from to resume computations.",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        default=[
            "generate_seqs",
            "compute_probs",
            "run_mcmc",
            "run_eval_mauve",
            "run_eval_bleu",
        ],
        choices=[
            "generate_seqs",
            "compute_probs",
            "run_mcmc",
            "run_eval_mauve",
            "run_eval_bleu",
        ],
        help="Specify which actions to perform. Defaults to all actions.",
    )

    # MCMC arguments
    parser.add_argument(
        "--mcmc_num_samples",
        type=int,
        default=100,
        help="Number of MCMC samples to generate.",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval_dataset_name",
        type=str,
        default="webtext",
        choices=[
            "webtext",
            "small-117M",
            "small-117M-k40",
            "medium-345M",
            "medium-345M-k40",
            "large-762M",
            "large-762M-k40",
            "xl-1542M",
            "xl-1542M-k40",
        ],
        help="Name of the dataset to use as reference.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Split of the dataset to use as reference.",
    )
    parser.add_argument(
        "--eval_num_sequences",
        type=int,
        default=None,
        help="Number of sequences to evaluate. If not provided, --num_mcmc_samples will be used.",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the output files.",
    )

    args = parser.parse_args()
    validate_args(args)

    return args


def get_output_subdir(args):
    # Generate a unique 6-character alphanumeric string
    unique_name = secrets.token_hex(3)  # Generates 6 hex characters

    if args.preload_dir:
        # Use the preload directory as the output directory
        output_subdir = os.path.join(args.output_dir, args.model_name, args.preload_dir)
    else:
        output_subdir = os.path.join(args.output_dir, args.model_name, unique_name)

    return output_subdir


def set_args_from_metadata(args, output_subdir):
    metadata_file_path = os.path.join(output_subdir, "metadata")
    metadata = load_from_json(metadata_file_path)

    # Load args from metadata json
    for key, value in metadata.items():
        # The parameters below are loaded from the command line and should not be overwritten
        if key != "preload_dir" and key != "actions" and key != "model_name":
            setattr(args, key, value)


def main():
    args = parse_args()

    output_subdir = get_output_subdir(args)

    if args.preload_dir:
        # Load metadata and parse arguments from it
        logging.info(f"Loading metadata from {output_subdir} as args...")
        set_args_from_metadata(args, output_subdir)

    logging.info(f"Args: {args}")
    # Create a directory to save the output files
    os.makedirs(output_subdir, exist_ok=True)
    # Save log messages to a file
    setup_logging(log_file=os.path.join(output_subdir, "log.txt"))
    # Save command-line arguments to JSON
    save_args(args, output_subdir)
    # Set the random seed for reproducibility
    set_seed(args.seed)

    (
        sequences_ids,
        sequences_decoded,
        target_logprobs,
        proposal_logprobs,
    ) = generate_sequences_and_probs(
        args, output_subdir=os.path.join(output_subdir, "sequences")
    )

    _, sampled_sequences_decoded, _ = run_mcmc(
        args=args,
        output_subdir=os.path.join(output_subdir, "mcmc"),
        sequences_ids=sequences_ids,
        sequences_decoded=sequences_decoded,
        target_logprobs=target_logprobs,  # target_logpropbs are probabilities sampled from the global unnormalized distribution
        proposal_logprobs=proposal_logprobs,  # proposal_logprobs are probabilities sampled from the local normalized distribution
    )

    _, _, _, _ = evaluate(
        args,
        output_subdir=os.path.join(output_subdir, "eval"),
        local_decoding_texts=sequences_decoded,  # sequences_decoded are the sequences sampled from the local normalized distribution
        global_decoding_texts=sampled_sequences_decoded,  # sampled_sequences_decoded are the sequences sampled from the global unnormalized distribution
    )


if __name__ == "__main__":
    main()
