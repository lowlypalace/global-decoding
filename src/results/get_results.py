import json
import os
import random
import logging
import pandas as pd
import numpy as np
from scipy import stats


def filter_padding_tokens(sequence):
    """Helper function to filter out padding tokens (tokens with value 0)."""
    return [token for token in sequence if token != 0]


def compute_95ci(data):
    """Compute the 95% confidence interval for a list of numbers."""
    mean = np.mean(data)
    ci = stats.norm.interval(0.95, loc=mean, scale=stats.sem(data)) if len(data) > 1 else (mean, mean)
    return mean, ci


def load_json_file(file_path):
    """Helper function to load JSON data from a file."""
    with open(file_path, "r") as f:
        return json.load(f)


def process_sequences(mcmc_dir, sequences_dir):
    """Process sequences by filtering padding tokens and calculating average lengths."""
    avg_lengths_sequences, avg_lengths_mcmc = [], []

    for run_dir in os.listdir(mcmc_dir):
        if run_dir.isdigit():
            mcmc_data = load_json_file(os.path.join(mcmc_dir, run_dir, "sampled_sequences_ids.json"))
            sequences_data = load_json_file(os.path.join(sequences_dir, "sequences_ids.json"))

            # Sample random sequences and filter padding tokens
            random_sequences = random.sample(sequences_data, min(len(sequences_data), 200))
            filtered_sequences = [filter_padding_tokens(seq) for seq in random_sequences]
            filtered_mcmc = [filter_padding_tokens(seq) for seq in mcmc_data]

            # Calculate average lengths
            avg_lengths_sequences.append(
                np.mean([len(seq) for seq in filtered_sequences]) if filtered_sequences else 0
            )
            avg_lengths_mcmc.append(
                np.mean([len(seq) for seq in filtered_mcmc]) if filtered_mcmc else 0
            )

    return avg_lengths_sequences, avg_lengths_mcmc


def process_log_likelihoods(mcmc_dir):
    """Process log likelihoods from the sampled sequences."""
    log_likelihoods_global, log_likelihoods_local = [], []

    for run_dir in os.listdir(mcmc_dir):
        if run_dir.isdigit():
            target_logprobs = load_json_file(os.path.join(mcmc_dir, run_dir, "sampled_target_logprobs.json"))
            proposal_logprobs = load_json_file(os.path.join(mcmc_dir, run_dir, "sampled_proposal_logprobs.json"))

            log_likelihoods_global.append(np.mean(target_logprobs))
            log_likelihoods_local.append(np.mean(proposal_logprobs))

    return log_likelihoods_global, log_likelihoods_local


def process_example_sequences(sequences_dir, mcmc_dir):
    """Process and decode example sequences."""
    sequences_decoded = load_json_file(os.path.join(sequences_dir, "sequences_decoded.json"))
    sampled_sequences_decoded = load_json_file(os.path.join(mcmc_dir, "0", "sampled_sequences_decoded.json"))

    # Sample a sequence and format
    sequence_decoded = random.sample(sequences_decoded, 1)[0][:100].replace("\n", "\\n")
    sequence_decoded_sampled = random.sample(sampled_sequences_decoded, 1)[0][:100].replace("\n", "\\n")

    return sequence_decoded, sequence_decoded_sampled


def load_constants(sequences_dir):
    """Load normalization constants, if available."""
    constants_file = os.path.join(sequences_dir, "proposal_normalize_constants_products.json")
    if os.path.exists(constants_file):
        return load_json_file(constants_file)
    return ""


def process_sub_directory(base_dir, sub_dir, model_name):
    """Process a single sub-directory and extract relevant results."""
    sequences_dir = os.path.join(base_dir, sub_dir, "sequences")
    mcmc_dir = os.path.join(base_dir, sub_dir, "mcmc")
    eval_dir = os.path.join(base_dir, sub_dir, "eval")

    if not (os.path.exists(sequences_dir) and os.path.exists(mcmc_dir)):
        return None

    # Load metadata and results
    metadata = load_json_file(os.path.join(base_dir, sub_dir, "metadata.json"))
    top_k, top_p = metadata.get("top_k", 50432), metadata.get("top_p", 1.0)

    logging.info(f"Model: {model_name}, Sub-directory: {sub_dir}, Top-k: {top_k}, Top-p: {top_p}")

    eval_results = load_json_file(os.path.join(eval_dir, "results.json"))

    # Process sequences and log likelihoods
    # avg_lengths_sequences, avg_lengths_mcmc = process_sequences(mcmc_dir, sequences_dir)
    # avg_length_sequences_mean, avg_length_sequences_ci = compute_95ci(avg_lengths_sequences)
    # avg_length_mcmc_mean, avg_length_mcmc_ci = compute_95ci(avg_lengths_mcmc)

    # log_likelihoods_global, log_likelihoods_local = process_log_likelihoods(mcmc_dir)
    # log_likelihoods_global_mean, log_likelihoods_global_ci = compute_95ci(log_likelihoods_global)
    # log_likelihoods_local_mean, log_likelihoods_local_ci = compute_95ci(log_likelihoods_local)

    # Process example sequences and constants
    sequence_decoded, sequence_decoded_sampled = process_example_sequences(sequences_dir, mcmc_dir)
    constants_products = load_constants(sequences_dir)

    return {
        "sub_dir": sub_dir,
        "top_k": top_k,
        "top_p": top_p,
        "mauve_local_mean": eval_results["mauve_local"]["mean"],
        "mauve_global_mean": eval_results["mauve_global"]["mean"],
        "bleu_local_mean": eval_results["bleu_local"]["mean"],
        "bleu_global_mean": eval_results["bleu_global"]["mean"],
        "mauve_local_ci": eval_results["mauve_local"]["ci"],
        "mauve_global_ci": eval_results["mauve_global"]["ci"],
        "bleu_local_ci": eval_results["bleu_local"]["ci"],
        "bleu_global_ci": eval_results["bleu_global"]["ci"],
        # "avg_length_local_mean": avg_length_sequences_mean,
        # "avg_length_local_ci": avg_length_sequences_ci,
        # "avg_length_global_mean": avg_length_mcmc_mean,
        # "avg_length_global_ci": avg_length_mcmc_ci,
        # "log_likelihoods_global_mean": log_likelihoods_global_mean,
        # "log_likelihoods_global_ci": log_likelihoods_global_ci,
        # "log_likelihoods_local_mean": log_likelihoods_local_mean,
        # "log_likelihoods_local_ci": log_likelihoods_local_ci,
        "sequence_local": sequence_decoded,
        "sequence_global": sequence_decoded_sampled,
        "constants_products": constants_products,
    }


def get_results(model_name):
    """Generate and return results for the specified model."""
    base_dir = os.path.join("output", model_name)
    results = []

    logging.info(f"Generating {model_name} results...")

    for sub_dir in os.listdir(base_dir):
        result = process_sub_directory(base_dir, sub_dir, model_name)
        if result:
            results.append(result)

    results_df = pd.DataFrame(results)

    # Sort and return results
    top_k_df = results_df.dropna(subset=["top_k"]).sort_values(by="top_k")
    top_p_df = results_df.dropna(subset=["top_p"]).sort_values(by="top_p")

    return top_k_df, top_p_df
