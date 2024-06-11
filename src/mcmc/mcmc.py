import os
import logging

from src.utils.utils import timer, save_to_json, load_from_json

from src.mcmc.metropolis_hastings import metropolis_hastings
from src.mcmc.plots import plot_distribution, plot_chain, plot_logprob_diff


def load_sampled_sequences(output_subdir):
    sampled_sequences_ids = load_from_json(
        os.path.join(output_subdir, "sampled_sequences_ids")
    )
    sampled_sequences_decoded = load_from_json(
        os.path.join(output_subdir, "sampled_sequences_decoded")
    )
    sampled_target_logprobs = load_from_json(
        os.path.join(output_subdir, "sampled_target_logprobs")
    )
    return sampled_sequences_ids, sampled_sequences_decoded, sampled_target_logprobs


def run_multiple_mh(
    args,
    output_subdir,
    sequences_ids,
    sequences_decoded,
    target_logprobs,
    proposal_logprobs,
):
    num_samples = args.mcmc_num_samples

    # Calculate the number of sequences per subset
    subset_size = len(sequences_ids) // num_samples
    logging.info(f"Number of sequences for each MCMC iteration: {subset_size}")

    sampled_sequences_ids = []
    sampled_sequences_decoded = []
    sampled_target_logprobs = []

    # Run the Independent Metropolis-Hastings algorithm
    with timer("Running MCMC algorithm"):
        for i in range(num_samples):
            start_idx = i * subset_size
            end_idx = (
                (i + 1) * subset_size if (i + 1) < num_samples else len(sequences_ids)
            )

            subset_sequences_ids = sequences_ids[start_idx:end_idx]
            subset_sequences_decoded = sequences_decoded[start_idx:end_idx]
            subset_target_logprobs = target_logprobs[start_idx:end_idx]
            subset_proposal_logprobs = proposal_logprobs[start_idx:end_idx]

            (
                collected_sequences_ids,
                collected_sequences_decoded,
                collected_target_logprobs,
                logprob_diff_proposed,
                logprob_diff_current,
                sequence_change_indices,
            ) = metropolis_hastings(
                sequence_count=subset_size,
                sequences_ids=subset_sequences_ids,
                sequences_decoded=subset_sequences_decoded,
                target_logprobs=subset_target_logprobs,
                proposal_logprobs=subset_proposal_logprobs,
            )

            # Save the sequences and their probabilities to JSON files
            save_to_json(
                collected_sequences_ids,
                "collected_sequences_ids",
                os.path.join(output_subdir, "plots", "runs", f"run_{i}"),
            )
            save_to_json(
                collected_sequences_decoded,
                "collected_sequences_decoded",
                os.path.join(output_subdir, "plots", "runs", f"run_{i}"),
            )
            save_to_json(
                collected_target_logprobs,
                "collected_target_logprobs",
                os.path.join(output_subdir, "plots", "runs", f"run_{i}"),
            )

            # Plot the chain of generated samples
            plot_chain(
                collected_target_logprobs,
                prefix="mcmc",
                show=False,
                output_dir=os.path.join(output_subdir, "plots", "runs", f"run_{i}"),
            )
            # Plot the deltas for the acceptance ratio
            plot_logprob_diff(
                logprob_diff_proposed,
                logprob_diff_current,
                sequence_change_indices,
                prefix="mcmc",
                show=False,
                output_dir=os.path.join(output_subdir, "plots", "runs", f"run_{i}"),
            )

            # Take the last sample from each Metropolis iteration and add it to the sampled sequences arrays
            sampled_sequences_ids.append(collected_sequences_ids[-1])
            sampled_sequences_decoded.append(collected_sequences_decoded[-1])
            sampled_target_logprobs.append(collected_target_logprobs[-1])

    return sampled_sequences_ids, sampled_sequences_decoded, sampled_target_logprobs


def run_mcmc(
    args,
    output_subdir,
    sequences_ids,
    sequences_decoded,
    target_logprobs,
    proposal_logprobs,
):

    if "run_mcmc" in args.actions:
        logging.info("Running the MCMC algorithm...")
        sampled_sequences_ids, sampled_sequences_decoded, sampled_target_logprobs = (
            run_multiple_mh(
                args,
                output_subdir,
                sequences_ids,
                sequences_decoded,
                target_logprobs,
                proposal_logprobs,
            )
        )

        logging.info(
            f"Sampled {len(sampled_sequences_ids)} sequences from the MCMC algorithm."
        )
        # Save the sampled sequences and their probabilities to JSON files
        save_to_json(sampled_sequences_ids, "sampled_sequences_ids", output_subdir)
        save_to_json(
            sampled_sequences_decoded, "sampled_sequences_decoded", output_subdir
        )
        save_to_json(sampled_target_logprobs, "sampled_target_logprobs", output_subdir)

        # Plot the distribution of the generated probabilities
        logging.info("Plotting the distribution of the sampled sequences...")
        plot_distribution(
            sampled_target_logprobs,
            plot_type="histogram",
            prefix="mcmc",
            show=False,
            output_dir=os.path.join(output_subdir, "plots"),
        )
        plot_distribution(
            sampled_target_logprobs,
            plot_type="kde",
            prefix="mcmc",
            show=False,
            output_dir=os.path.join(output_subdir, "plots"),
        )

        return sampled_sequences_ids, sampled_sequences_decoded, sampled_target_logprobs

    elif "run_eval_mauve" in args.actions or "run_eval_bleu" in args.actions:
        logging.info("Loading precomputed MCMC samples...")
        sampled_sequences_ids, sampled_sequences_decoded, sampled_target_logprobs = (
            load_sampled_sequences(output_subdir)
        )
        return sampled_sequences_ids, sampled_sequences_decoded, sampled_target_logprobs
