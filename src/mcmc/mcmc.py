import numpy as np
import os

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

    # Run the Independent Metropolis-Hastings algorithm
    with timer("Running MCMC algorithm"):
        (
            sampled_sequences_ids,
            sampled_sequences_decoded,
            sampled_target_logprobs,
            logprob_diff_proposed,
            logprob_diff_current,
            sequence_change_indices,
        ) = metropolis_hastings(
            sequence_count=sequence_count,
            burnin=burnin,
            sequences_ids=sequences_ids,
            sequences_decoded=sequences_decoded,
            target_logprobs=target_logprobs,
            proposal_logprobs=proposal_logprobs,
            sample_rate=sample_rate,
        )

    # Save the sampled sequences and their probabilities to JSON files
    save_to_json(sampled_sequences_ids, "sampled_sequences_ids", output_subdir)
    save_to_json(sampled_sequences_decoded, "sampled_sequences_decoded", output_subdir)
    save_to_json(sampled_target_logprobs, "sampled_target_logprobs", output_subdir)

    # Plot the distribution of the generated probabilities
    with timer("Plotting the results"):
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
        # Plot the chain of generated samples
        plot_chain(
            sampled_target_logprobs,
            burnin=burnin,
            show=False,
            output_dir=os.path.join(output_subdir, "plots"),
        )
        # Plot the deltas for the acceptance ratio
        plot_logprob_diff(
            logprob_diff_proposed,
            logprob_diff_current,
            sequence_change_indices,
            show=False,
            output_dir=os.path.join(output_subdir, "plots"),
        )

    return sampled_sequences_ids, sampled_sequences_decoded, sampled_target_logprobs
