import os
import logging

from utils import timer, save_to_json

from .metropolis_hastings import metropolis_hastings
from .plots import plot_distribution, plot_chain, plot_logprob_diff


def run_mcmc(
    args,
    output_subdir,
    sequences_ids,
    sequences_decoded,
    target_logprobs,
    proposal_logprobs,
):
    # sequence_count = args.sequence_count
    num_subsets = args.mcmc_num_subsets

    # Calculate the number of sequences per subset
    subset_size = len(sequences_ids) // num_subsets

    sampled_sequences_ids = []
    sampled_sequences_decoded = []
    sampled_target_logprobs = []

    # Run the Independent Metropolis-Hastings algorithm
    with timer("Running MCMC algorithm"):
        for i in range(num_subsets):
            start_idx = i * subset_size
            end_idx = (i + 1) * subset_size if (i + 1) < num_subsets else len(sequences_ids)

            subset_sequences_ids = sequences_ids[start_idx:end_idx]
            subset_sequences_decoded = sequences_decoded[start_idx:end_idx]
            subset_target_logprobs = target_logprobs[start_idx:end_idx]
            subset_proposal_logprobs = proposal_logprobs[start_idx:end_idx]

            logging.info(f"Running MCMC algorithm on subset {i + 1} of {num_subsets}: {start_idx} - {end_idx}...")

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
                output_dir=os.path.join(
                    output_subdir, "plots", "runs", f"run_{i}"
                ),
            )
            # Plot the deltas for the acceptance ratio
            plot_logprob_diff(
                logprob_diff_proposed,
                logprob_diff_current,
                sequence_change_indices,
                prefix="mcmc",
                show=False,
                output_dir=os.path.join(
                    output_subdir, "plots", "runs", f"run_{i}"
                ),
            )

            # Take the last sample from each Metropolis iteration and add it to the sampled sequences arrays
            sampled_sequences_ids.append(collected_sequences_ids[-1])
            sampled_sequences_decoded.append(collected_sequences_decoded[-1])
            sampled_target_logprobs.append(collected_target_logprobs[-1])

    # Save the sampled sequences and their probabilities to JSON files
    save_to_json(sampled_sequences_ids, "sampled_sequences_ids", output_subdir)
    save_to_json(sampled_sequences_decoded, "sampled_sequences_decoded", output_subdir)
    save_to_json(sampled_target_logprobs, "sampled_target_logprobs", output_subdir)

    # Plot the distribution of the generated probabilities
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
