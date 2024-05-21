import os
import numpy as np

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
    # Parse command-line arguments
    burnin = args.mcmc_burnin
    sample_rate = args.mcmc_sample_rate
    sequence_count = args.sequence_count
    seed = args.seed

    # Calculate how many independent runs are needed
    independent_runs = sequence_count // sample_rate

    sampled_sequences_ids = []
    sampled_sequences_decoded = []
    sampled_target_logprobs = []

    # Run the Independent Metropolis-Hastings algorithm
    with timer("Running MCMC algorithm"):
        for i in range(independent_runs):

            (
                collected_sequences_ids,
                collected_sequences_decoded,
                collected_target_logprobs,
                logprob_diff_proposed,
                logprob_diff_current,
                sequence_change_indices,
            ) = metropolis_hastings(
                sequence_count=sequence_count,
                sequences_ids=sequences_ids,
                sequences_decoded=sequences_decoded,
                target_logprobs=target_logprobs,
                proposal_logprobs=proposal_logprobs,
            )

            # Save the sequences and their probabilities to JSON files
            save_to_json(
                collected_sequences_ids,
                f"collected_sequences_ids",
                os.path.join(output_subdir, "plots", "independent_runs", f"run_{i}"),
            )
            save_to_json(
                collected_sequences_decoded,
                f"collected_sequences_decoded",
                os.path.join(output_subdir, "plots", "independent_runs" , f"run_{i}"),
            )
            save_to_json(
                collected_target_logprobs,
                f"collected_target_logprobs",
                os.path.join(output_subdir, "plots", "independent_runs", f"run_{i}"),
            )

            # Plot the chain of generated samples
            plot_chain(
                collected_target_logprobs,
                burnin=burnin,
                prefix=f"mcmc",
                show=False,
                output_dir=os.path.join(output_subdir, "plots", "independent_runs", f"run_{i}"),
            )
            # Plot the deltas for the acceptance ratio
            plot_logprob_diff(
                logprob_diff_proposed,
                logprob_diff_current,
                sequence_change_indices,
                prefix=f"mcmc",
                show=False,
                output_dir=os.path.join(output_subdir, "plots", "independent_runs", f"run_{i}"),
            )

            # Take the last sample from each Metropolis iteration and add it to the sampled sequences arrays
            sampled_sequences_ids.append(sequences_ids[-1])
            sampled_sequences_decoded.append(sequences_decoded[-1])
            sampled_target_logprobs.append(target_logprobs[-1])

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
