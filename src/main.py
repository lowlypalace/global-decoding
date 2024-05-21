import torch
import argparse
import torch
import os


from utils import setup_logging, save_args, get_timestamp, validate_args, set_seed
from sequences import generate_sequences_and_probs
from mcmc import run_mcmc
from eval import evaluate


# Define the function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text sequences, run MCMC, and evaluate the results."
    )

    # Sequence generation arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        choices=[
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "pythia-70m",
            "pythia-160m",
            "pythia-410m",
            "pythia-1b", # maybe drop and keep 1.4b
            "pythia-1.4b",
            "pythia-2.8b",
            "pythia-6.9b",
            "pythia-12b", # ommited if not enough time
        ],
        help="Model to use for text generation. Supports GPT-2 and Pythia.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to use as a prompt. Defaults to the EOS token.",
    )
    decoding_group = (
        parser.add_mutually_exclusive_group()
    )  # Mutually exclusive group for top-k and top-p
    decoding_group.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Top-k value for text generation. Defaults to 100 if neither top-k nor top-p is provided.",
    )
    decoding_group.add_argument(
        "--top_p",
        type=float,
        help="Top-p value for text generation. No default value; must be specified if used.",
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

    # MCMC arguments
    parser.add_argument(
        "--mcmc_num_subsets",
        type=int,
        default=10,
        help="Number of subsets to split the sequences into for MCMC processing.",
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
        help="Number of sequences to evaluate. If not provided, (1 - burnin) * sequence_count * sample_rate * 0.01 sequences will be evaluated.",
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


def main():
    args = parse_args()

    # Add a directory with a timestamp to the output directory
    output_subdir = os.path.join(args.output_dir, get_timestamp())
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

    sampled_sequences_ids, sampled_sequences_decoded, sampled_logprobs = run_mcmc(
        args=args,
        output_subdir=os.path.join(output_subdir, "mcmc"),
        sequences_ids=sequences_ids,
        sequences_decoded=sequences_decoded,
        target_logprobs=target_logprobs,  # target_logpropbs are probabilities sampled from the global unnormalized distribution
        proposal_logprobs=proposal_logprobs,  # proposal_logprobs are probabilities sampled from the local normalized distribution
    )

    mauve_results_local, mauve_results_global = evaluate(
        args,
        output_subdir=os.path.join(output_subdir, "eval"),
        local_decoding_texts=sequences_decoded,  # sequences_decoded are the sequences sampled from the local normalized distribution
        global_decoding_texts=sampled_sequences_decoded,  # sampled_sequences_decoded are the sequences sampled from the global unnormalized distribution
    )


if __name__ == "__main__":
    main()
