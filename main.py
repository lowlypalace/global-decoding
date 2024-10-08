import torch
import argparse
import logging
import secrets
import os
import random
from scipy import stats
import numpy as np

from src.utils.utils import (
    setup_logging,
    save_args,
    set_seed,
    load_from_json,
    save_to_json,
    convert_tensor_to_list,
    get_unique_name,
)
from src.utils.validate import validate_args


from src.sequences import generate_sequences, compute_probs, prune_sequences
from src.mcmc import run_mcmc
from src.eval import evaluate


class NonDefaultAction(argparse.Action):
    """Custom action to track if an argument was provided via command line."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_nondefault", True)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate text sequences, run MCMC, and evaluate the results.")

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
        action=NonDefaultAction,
        help="Model to use for text generation. Supports GPT-2 and Pythia.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        action=NonDefaultAction,
        help="Text to use as a prompt. Defaults to the EOS token.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        action=NonDefaultAction,
        help="Top-k value for text generation. No default value; must be specified if used.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        action=NonDefaultAction,
        help="Top-p value for text generation. No default value; must be specified if used.",
    )
    parser.add_argument(
        "--sequence_count",
        type=int,
        default=1000,
        action=NonDefaultAction,
        help="Number of sequence samples to generate and use for MCMC analysis.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        action=NonDefaultAction,
        help="Maximum sequence length. If not provided, it will be set to the maximum model length minus the length of the input text.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        action=NonDefaultAction,
        help='Device to use for computation. Defaults to "cuda" if available.',
    )
    parser.add_argument(
        "--batch_size_seq",
        type=int,
        default=64,
        action=NonDefaultAction,
        help="Batch size for generating sequences.",
    )
    parser.add_argument(
        "--batch_size_prob",
        type=int,
        default=16,
        action=NonDefaultAction,
        help="Batch size for computing probabilities.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp64",
        choices=["fp16", "fp32", "fp64"],
        action=NonDefaultAction,
        help="Precision to use for the model. Defaults to fp64.",
    )

    parser.add_argument(
        "--preload_dir",
        type=str,
        default=None,
        action=NonDefaultAction,
        help="Directory name to preload generated sequences from to resume computations.",
    )

    # MCMC arguments
    parser.add_argument(
        "--mcmc_num_samples",
        type=int,
        default=200,
        help="Number of MCMC samples to generate.",
    )
    parser.add_argument(
        "--mcmc_num_sequences",
        type=int,
        default=None,
        action=NonDefaultAction,
        help="Number of sequences to consider for each MCMC chain. If not provided, --sequence count / --mcmc_num_samples will be used.",
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
        action=NonDefaultAction,
        help="Name of the dataset to use as reference.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        action=NonDefaultAction,
        help="Split of the dataset to use as reference.",
    )
    parser.add_argument(
        "--eval_num_sequences",
        type=int,
        default=None,
        action=NonDefaultAction,
        help="Number of sequences to evaluate. If not provided, --num_mcmc_samples will be evaluated.",
    )
    parser.add_argument(
        "--eval_num_runs",
        type=int,
        default=10,
        action=NonDefaultAction,
        help="Number of runs for MAUVE and BLEU evaluations.",
    )

    # Other arguments
    parser.add_argument(
        "--actions",
        nargs="+",
        default=["generate_seqs", "compute_probs", "run_mcmc_eval"],
        choices=["generate_seqs", "compute_probs", "run_mcmc_eval"],
        action=NonDefaultAction,
        help="Specify which actions to perform. Defaults to all actions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        action=NonDefaultAction,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        action=NonDefaultAction,
        help="Directory to save the output files.",
    )

    args = parser.parse_args()
    validate_args(args)

    return args


def create_output_subdir(args):
    """Creates the output directory."""
    subdir = args.preload_dir or get_unique_name()
    # TODO: REMOVE "merged"
    output_subdir = os.path.join(args.output_dir, "merged", args.model_name, subdir)
    os.makedirs(output_subdir, exist_ok=True)
    return output_subdir


def load_metadata(args, output_subdir):
    """Loads metadata from a previously saved run and sets args from it."""
    logging.info(f"Loading metadata from {output_subdir} as args...")
    metadata = load_from_json(os.path.join(output_subdir, "metadata"))
    for key, value in metadata.items():
        if not hasattr(args, f"{key}_nondefault"):
            setattr(args, key, value)


def calculate_stats(scores):
    """Calculates the mean and confidence interval for a given list of scores."""
    mean = np.mean(scores)
    ci = stats.norm.interval(0.95, loc=mean, scale=stats.sem(scores))
    return {"mean": mean, "ci": ci, "scores": scores}


def init_run(args, run_idx):
    """Initialize a run with a specific seed."""
    seed = args.seed + run_idx
    set_seed(seed)
    logging.info(f"Starting run {run_idx + 1}/{args.eval_num_runs} with seed {seed}")
    return seed


def load_sequences(output_subdir, device):
    """Load preloaded sequences."""
    logging.info("Loading preloaded sequences...")
    sequences_ids = load_from_json(os.path.join(output_subdir, "sequences_ids"))
    sequences_decoded = load_from_json(os.path.join(output_subdir, "sequences_decoded"))
    return torch.tensor(sequences_ids).to(device), sequences_decoded


def load_probs(output_subdir, device):
    """Load precomputed probabilities."""
    logging.info("Loading precomputed probabilities...")
    return (
        torch.tensor(load_from_json(os.path.join(output_subdir, "logprobs_target"))).to(device),
        torch.tensor(load_from_json(os.path.join(output_subdir, "logprobs_proposal"))).to(device),
    )


def main():
    args = parse_args()
    output_subdir = create_output_subdir(args)
    setup_logging(log_file=os.path.join(output_subdir, "log.txt"))

    if args.preload_dir:
        load_metadata(args, output_subdir)

    logging.info(f"Args: {args}")
    save_args(args, output_subdir)
    set_seed(args.seed)

    output_subdir_seqs = os.path.join(output_subdir, "sequences")
    output_subdir_mcmc = os.path.join(output_subdir, "mcmc")
    # TODO: Rename to probs
    output_subdir_probs = os.path.join(output_subdir, "sequences")
    output_subdir_eval = os.path.join(output_subdir, "eval")

    mauve_scores_local, mauve_scores_global = [], []
    bleu_scores_local, bleu_scores_global = [], []

    eval_num_sequences = args.eval_num_sequences or args.mcmc_num_samples

    if "generate_seqs" in args.actions:
        sequences_ids, sequences_decoded = generate_sequences(args, output_subdir_seqs)
    else:
        sequences_ids, sequences_decoded = load_sequences(output_subdir_seqs, args.device)

    if "compute_probs" in args.actions:
        target_logprobs, proposal_logprobs = compute_probs(args, sequences_ids, output_subdir_probs)
    else:
        target_logprobs, proposal_logprobs = load_probs(output_subdir_probs, args.device)

    if "run_mcmc_eval" in args.actions:
        sequences_ids = convert_tensor_to_list(sequences_ids)

        # Prune sequences
        sequences_ids, sequences_decoded, target_logprobs, proposal_logprobs = prune_sequences(
            args, sequences_ids, sequences_decoded, target_logprobs, proposal_logprobs
        )

        for run_idx in range(args.eval_num_runs):
            seed = init_run(args, run_idx)

            # Bootstrapping: generate indices to select elements for all arrays
            bootstrap_indices = random.choices(range(len(sequences_decoded)), k=len(sequences_decoded))
            bootstrapped_sequences_decoded = [sequences_decoded[i] for i in bootstrap_indices]
            bootstrapped_sequences_ids = [sequences_ids[i] for i in bootstrap_indices]
            bootstrapped_target_logprobs = [target_logprobs[i] for i in bootstrap_indices]
            bootstrapped_proposal_logprobs = [proposal_logprobs[i] for i in bootstrap_indices]

            _, sampled_sequences_decoded, _ = run_mcmc(
                args=args,
                output_subdir=os.path.join(output_subdir_mcmc, str(seed)),
                sequences_ids=bootstrapped_sequences_ids,
                sequences_decoded=bootstrapped_sequences_decoded,
                target_logprobs=bootstrapped_target_logprobs,  # Probabilities sampled from the global unnormalized distribution
                proposal_logprobs=bootstrapped_proposal_logprobs,  # Probabilities sampled from the local normalized distribution
            )

            eval_local_decoding_texts = bootstrapped_sequences_decoded[:eval_num_sequences]
            eval_global_decoding_texts = sampled_sequences_decoded[:eval_num_sequences]

            mauve_results_local, mauve_results_global, bleu_local, bleu_global = evaluate(
                args,
                output_subdir=os.path.join(output_subdir_eval, str(seed)),
                eval_local_decoding_texts=eval_local_decoding_texts,  # Sequences sampled from the local normalized distribution
                eval_global_decoding_texts=eval_global_decoding_texts,  # Sequences sampled from the global unnormalized distribution
                eval_num_sequences=eval_num_sequences,
                seed=seed,
            )

            # Accumulate scores
            mauve_scores_local.append(mauve_results_local.mauve)
            mauve_scores_global.append(mauve_results_global.mauve)
            bleu_scores_local.append(bleu_local)
            bleu_scores_global.append(bleu_global)

        results = {
            "mauve_local": calculate_stats(mauve_scores_local),
            "mauve_global": calculate_stats(mauve_scores_global),
            "bleu_local": calculate_stats(bleu_scores_local),
            "bleu_global": calculate_stats(bleu_scores_global),
        }

        save_to_json(results, "results", output_subdir_eval)
        logging.info(f"Results saved: {results}")


if __name__ == "__main__":
    main()
