import json
import os
import random
import logging
import pandas as pd


def filter_padding_tokens(sequence):
    """Helper function to filter out padding tokens (tokens with value 0)."""
    return [token for token in sequence if token != 0]


def get_results(model_name):
    base_dir = os.path.join("output", model_name)
    results = []

    logging.info(f"Generating {model_name} results...")

    for sub_dir in os.listdir(base_dir):
        sequences_dir = os.path.join(base_dir, sub_dir, "sequences")
        mcmc_dir = os.path.join(base_dir, sub_dir, "mcmc", "0")
        eval_dir = os.path.join(base_dir, sub_dir, "eval")

        if os.path.exists(sequences_dir) and os.path.exists(mcmc_dir):
            try:
                # Load metadata
                with open(os.path.join(base_dir, sub_dir, "metadata.json"), "r") as f:
                    metadata = json.load(f)
                top_k = metadata.get("top_k")
                top_p = metadata.get("top_p")

                # when top_k and top_p are equal to None
                if top_k is None and top_p is None:
                    top_k = 50432
                    top_p = 1.0

                ##################
                # Eval Results (MAUVE, BLEU)
                ##################
                with open(os.path.join(eval_dir, "results.json"), "r") as f:
                    eval_results = json.load(f)

                mauve_local_mean = eval_results["mauve_local"]["mean"]
                mauve_global_mean = eval_results["mauve_global"]["mean"]
                bleu_local_mean = eval_results["bleu_local"]["mean"]
                bleu_global_mean = eval_results["bleu_global"]["mean"]

                mauve_local_ci = eval_results["mauve_local"]["ci"]
                mauve_global_ci = eval_results["mauve_global"]["ci"]
                bleu_local_ci = eval_results["bleu_local"]["ci"]
                bleu_global_ci = eval_results["bleu_global"]["ci"]

                mauve_local_scores = eval_results["mauve_local"]["scores"]
                mauve_global_scores = eval_results["mauve_global"]["scores"]
                bleu_local_scores = eval_results["bleu_local"]["scores"]
                bleu_global_scores = eval_results["bleu_global"]["scores"]

                ##################
                # Sequences Lengths
                ##################

                with open(os.path.join(sequences_dir, "sequences_ids.json"), "r") as f:
                    sequences_data = json.load(f)

                with open(os.path.join(mcmc_dir, "sampled_sequences_ids.json"), "r") as f:
                    mcmc_data = json.load(f)

                # Sample first 200 random sequences
                random_sequences = random.sample(sequences_data, min(len(sequences_data), 200))

                # Filter padding tokens and compute average lengths
                filtered_sequences = [filter_padding_tokens(seq) for seq in random_sequences]
                filtered_mcmc = [filter_padding_tokens(seq) for seq in mcmc_data]

                avg_length_sequences = (
                    sum(len(seq) for seq in filtered_sequences) / len(filtered_sequences) if filtered_sequences else 0
                )
                avg_length_mcmc = sum(len(seq) for seq in filtered_mcmc) / len(filtered_mcmc) if filtered_mcmc else 0

                ##################
                # Example Sequences
                ##################

                with open(os.path.join(sequences_dir, "sequences_decoded.json"), "r") as f:
                    sequences_decoded = json.load(f)
                sequence_decoded = random.sample(sequences_decoded, 1)[0][:100]
                sequence_decoded = sequence_decoded.replace("\n", "\\n")

                with open(os.path.join(mcmc_dir, "sampled_sequences_decoded.json"), "r") as f:
                    sampled_sequences_decoded = json.load(f)
                sequence_decoded_sampled = random.sample(sampled_sequences_decoded, 1)[0][:100]
                sequence_decoded_sampled = sequence_decoded_sampled.replace("\n", "\\n")

                ###################
                # Log likelihood
                ###################
                with open(os.path.join(mcmc_dir, "sampled_target_logprobs.json"), "r") as f:
                    sampled_target_logprobs = json.load(f)

                log_likelihood_global = sum(sampled_target_logprobs) / len(sampled_target_logprobs)

                with open(os.path.join(mcmc_dir, "sampled_proposal_logprobs.json"), "r") as f:
                    sampled_proposal_logprobs = json.load(f)

                log_likelihood_local = sum(sampled_proposal_logprobs) / len(sampled_proposal_logprobs)

                ###################
                # Decoding constants:
                ###################
                constants_products = ""
                constants_file_path = os.path.join(sequences_dir, "proposal_normalize_constants_products.json")
                if os.path.exists(constants_file_path):
                    with open(constants_file_path, "r") as f:
                        constants_products = json.load(f)

                results.append(
                    {
                        "sub_dir": sub_dir,
                        "top_k": top_k,
                        "top_p": top_p,
                        "mauve_local_mean": mauve_local_mean,
                        "mauve_global_mean": mauve_global_mean,
                        "bleu_local_mean": bleu_local_mean,
                        "bleu_global_mean": bleu_global_mean,
                        "mauve_local_ci": mauve_local_ci,
                        "mauve_global_ci": mauve_global_ci,
                        "bleu_local_ci": bleu_local_ci,
                        "bleu_global_ci": bleu_global_ci,
                        "mauve_local_scores": mauve_local_scores,
                        "mauve_global_scores": mauve_global_scores,
                        "bleu_local_scores": bleu_local_scores,
                        "bleu_global_scores": bleu_global_scores,
                        "log_likelihood_local": log_likelihood_local,
                        "log_likelihood_global": log_likelihood_global,
                        "avg_length_local": avg_length_sequences,
                        "avg_length_global": avg_length_mcmc,
                        "sequence_local": sequence_decoded,
                        "sequence_global": sequence_decoded_sampled,
                        "constants_products": constants_products,
                    }
                )

            except Exception as e:
                print(f"Error processing {sub_dir}: {e}")

    results_df = pd.DataFrame(results)

    # Sort
    top_k_df = results_df.dropna(subset=["top_k"]).sort_values(by="top_k")
    top_p_df = results_df.dropna(subset=["top_p"]).sort_values(by="top_p")

    return top_k_df, top_p_df
