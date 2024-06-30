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
        mcmc_dir = os.path.join(base_dir, sub_dir, "mcmc")
        eval_dir = os.path.join(base_dir, sub_dir, "eval")

        if os.path.exists(sequences_dir) and os.path.exists(mcmc_dir):
            try:
                # Load metadata
                with open(os.path.join(base_dir, sub_dir, "metadata.json"), "r") as f:
                    metadata = json.load(f)
                top_k = metadata.get("top_k")
                top_p = metadata.get("top_p")

                # TODO: handle missing results by skipping (if metadata exists but some files not)

                # when top_k and top_p are equal to None
                if top_k is None and top_p is None:
                    top_k = 50432
                    top_p = 1.0

                ##################
                # MAUVE
                ##################

                with open(
                    os.path.join(eval_dir, "mauve_results_global.json"), "r"
                ) as f:
                    global_result = json.load(f)
                global_mauve = global_result.get("mauve")

                # Load local MAUVE result
                with open(os.path.join(eval_dir, "mauve_results_local.json"), "r") as f:
                    local_result = json.load(f)
                local_mauve = local_result.get("mauve")

                ##################
                # BLEU
                ##################

                with open(os.path.join(eval_dir, "self_bleu_results.json"), "r") as f:
                    bleu = json.load(f)
                local_bleu = bleu.get("local_self_bleu")
                global_bleu = bleu.get("global_self_bleu")

                ##################
                # Sequences Lengths
                ##################

                with open(os.path.join(sequences_dir, "sequences_ids.json"), "r") as f:
                    sequences_data = json.load(f)

                with open(
                    os.path.join(mcmc_dir, "sampled_sequences_ids.json"), "r"
                ) as f:
                    mcmc_data = json.load(f)

                # Sample first 200 random sequences
                random_sequences = random.sample(
                    sequences_data, min(len(sequences_data), 200)
                )

                # Filter padding tokens and compute average lengths
                filtered_sequences = [
                    filter_padding_tokens(seq) for seq in random_sequences
                ]
                filtered_mcmc = [filter_padding_tokens(seq) for seq in mcmc_data]

                avg_length_sequences = (
                    sum(len(seq) for seq in filtered_sequences)
                    / len(filtered_sequences)
                    if filtered_sequences
                    else 0
                )
                avg_length_mcmc = (
                    sum(len(seq) for seq in filtered_mcmc) / len(filtered_mcmc)
                    if filtered_mcmc
                    else 0
                )

                ##################
                # Example Sequences
                ##################

                with open(
                    os.path.join(sequences_dir, "sequences_decoded.json"), "r"
                ) as f:
                    sequences_decoded = json.load(f)
                sequence_decoded = random.sample(sequences_decoded, 1)[0][:100]
                sequence_decoded = sequence_decoded.replace("\n", "\\n")

                with open(
                    os.path.join(mcmc_dir, "sampled_sequences_decoded.json"), "r"
                ) as f:
                    sampled_sequences_decoded = json.load(f)
                sequence_decoded_sampled = random.sample(sampled_sequences_decoded, 1)[
                    0
                ][:100]
                sequence_decoded_sampled = sequence_decoded_sampled.replace("\n", "\\n")

                ###################
                # Log likelihood
                ###################
                with open(
                    os.path.join(mcmc_dir, "sampled_target_logprobs.json"), "r"
                ) as f:
                    sampled_target_logprobs = json.load(f)

                average_log_likelihood = sum(sampled_target_logprobs) / len(
                    sampled_target_logprobs
                )

                results.append(
                    {
                        "sub_dir": sub_dir,
                        "top_k": top_k,
                        "top_p": top_p,
                        "mauve_local": local_mauve,
                        "mauve_global": global_mauve,
                        "bleu_local": local_bleu,
                        "global_bleu": global_bleu,
                        "average_log_likelihood": average_log_likelihood,
                        "avg_length_local": avg_length_sequences,
                        "avg_length_global": avg_length_mcmc,
                        "sequence_local": sequence_decoded,
                        "sequence_global": sequence_decoded_sampled,
                    }
                )

                ###################
                # Decoding constants:
                # TODO
                # Compute product of local normalization constants
                ###################

            except Exception as e:
                print(f"Error processing {sub_dir}: {e}")

    results_df = pd.DataFrame(results)

    # Sort
    top_k_df = results_df.dropna(subset=["top_k"]).sort_values(by="top_k")
    top_p_df = results_df.dropna(subset=["top_p"]).sort_values(by="top_p")

    return top_k_df, top_p_df
