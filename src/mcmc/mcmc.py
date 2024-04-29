import numpy as np
import os

from metropolis_hastings import metropolis_hastings

from plots import plot_mcmc_distribution, plot_chain
from utils import timer


def run_mcmc(
    args,
    output_subdir,
    sequences_ids,
    sequences_decoded,
    target_logprobs,
    proposal_logprobs,
):
    # Parse command-line arguments
    burnin = args.burnin
    rate = args.rate
    seed = args.seed
    sequence_count = args.sequence_count

    # Set random seed for reproducibility numpy
    np.random.seed(seed)

    # Run the Independent Metropolis-Hastings algorithm
    with timer("Running MCMC algorithm"):
        (
            sampled_sequences,
            sampled_sequences_decoded,
            sampled_logprobs,
        ) = metropolis_hastings(
            sequence_count=sequence_count,
            burnin=burnin,
            sequences_ids=sequences_ids,
            sequences_decoded=sequences_decoded,
            target_logprobs=target_logprobs,
            proposal_logprobs=proposal_logprobs,
            rate=rate,
            output_subdir=output_subdir,
        )

    # Plot the distribution of the generated probabilities
    with timer("Plotting the results"):
        plot_mcmc_distribution(
            sampled_logprobs,
            plot_type="histogram",
            show=False,
            output_dir=os.path.join(output_subdir, "plots"),
        )
        plot_mcmc_distribution(
            sampled_logprobs,
            plot_type="kde",
            show=False,
            output_dir=os.path.join(output_subdir, "plots"),
        )
        # Plot the chain of generated samples
        plot_chain(
            sampled_logprobs,
            burnin=burnin,
            show=False,
            output_dir=os.path.join(output_subdir, "plots"),
        )

    return sampled_sequences, sampled_sequences_decoded, sampled_logprobs
