import torch
import argparse
import torch
import os


# from sequence_probability import get_sequence_probs
from utils import setup_logging, save_args, get_timestamp
from sequences import generate_sequences_and_probs
from mcmc import run_mcmc
from evaluate import evaluate


# Define the function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate text sequences.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        choices=[
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
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
    # Mutually exclusive group for top-k and top-p
    decoding_group = parser.add_mutually_exclusive_group()
    decoding_group.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Top-k value for text generation. Defaults to 100 if neither top-k nor top-p is provided."
    )
    decoding_group.add_argument(
        "--top_p",
        type=float,
        help="Top-p value for text generation. No default value; must be specified if used."
    )
    parser.add_argument(
        "--sequence_count",
        type=int,
        default=100,
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
    parser.add_argument(
        "--burnin",
        type=float,
        default=0.2,
        help="Burn-in period as a fraction of the total number of samples.",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=10,
        help="Rate at which to sample sequences after the burn-in period.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Add a directory with a timestamp to the output directory
    output_dir = os.path.join(args.output_dir, get_timestamp())
    # Create a directory to save the output files
    os.makedirs(output_dir, exist_ok=True)
    # Save log messages to a file
    setup_logging(log_file=os.path.join(output_dir, "log.txt"))
    # Save command-line arguments to JSON
    save_args(args, output_dir)

    # Generate sequences
    (
        sequences_ids,
        sequences_decoded,
        target_logprobs,
        proposal_logprobs,
    ) = generate_sequences_and_probs(
        args, output_subdir=os.path.join(output_dir, "sequences")
    )


    # MCMC
    sampled_sequences, sampled_sequences_decoded, sampled_logprobs = run_mcmc(
        args=args,
        output_subdir=os.path.join(output_dir, "mcmc"),
        sequences_ids=sequences_ids,
        sequences_decoded=sequences_decoded,
        target_logprobs=target_logprobs, # target_logpropbs are probabilities sampled from the global unnormalized distribution
        proposal_logprobs=proposal_logprobs, # proposal_logprobs are probabilities sampled from the local normalized distribution
    )

    # TODO: Evaluate

    # sequences_decoded are the sequences sampled from the local normalized distribution
    # sampled_sequences_decoded are the sequences sampled from the global unnormalized distribution

    evaluate(args, output_subdir=os.path.join(output_dir, "evaluate"), local_decoding_texts=sequences_decoded, global_decoding_texts=sampled_sequences_decoded)

    # TODO: get total time

if __name__ == "__main__":
    main()
