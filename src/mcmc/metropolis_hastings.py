import numpy as np
import json
import os

from utils import (
    create_filename,
)


from plots import plot_mcmc_distribution, plot_chain
from utils import timer


def indicator_top_k(sequence):
    # In our case, we can simply return 1 as we are using top-k sampling
    return 1


def metropolis_hastings(
    sequence_count,
    burnin,
    sequences_ids,
    sequences_decoded,
    target_logprobs,
    proposal_logprobs,
    rate,
    output_dir,
):
    # List to store the generated samples
    sampled_sequences = []
    sampled_decoded_sequences = []
    sampled_target_logprobs = []

    # Calculate the number of burn-in samples
    burnin_index = int(burnin * sequence_count)

    # Get the first sequence and its probabilities
    current_sequence = sequences_ids[0]
    current_decoded_seq = sequences_decoded[0]
    logprob_target_current, logprob_proposal_current = (
        target_logprobs[0],
        proposal_logprobs[0],
    )

    # This is a top-level loop to generate multiple sequences
    for i in range(1, sequence_count):
        # Get the sequence to propose
        proposed_sequence = sequences_ids[i]
        # Get the probabilities for the proposed sequences
        logprob_target_proposed, logprob_proposal_proposed = (
            # target_logprobs[i].item(),
            # proposal_logprobs[i].item(),
            target_logprobs[i],
            proposal_logprobs[i],
        )

        # Calculate the acceptance ratio
        numerator = (
            logprob_target_proposed
            + indicator_top_k(proposed_sequence)
            + logprob_proposal_current
        )
        denominator = (
            logprob_target_current
            + indicator_top_k(current_sequence)
            + logprob_proposal_proposed
        )
        log_acceptance_ratio = numerator - denominator

        # Accept or reject the new sequence based on the acceptance ratio
        if np.log(np.random.uniform(0, 1)) < log_acceptance_ratio:
            current_sequence = proposed_sequence
            logprob_target_current = logprob_target_proposed
            logprob_proposal_current = logprob_proposal_proposed

        # After the burn-in period, add the current state to the list of samples at the specified rate
        if i >= burnin_index and i % rate == 0:
            sampled_sequences.append(current_sequence)
            # Append the decoded sequence and its probabilities to samples
            sampled_decoded_sequences.append(current_decoded_seq)
            sampled_target_logprobs.append(logprob_target_current)

    with open(create_filename("sampled_sequences", "json", output_dir), "w") as f:
        json.dump(sampled_sequences, f)
    with open(
        create_filename("sampled_decoded_sequences", "json", output_dir), "w"
    ) as f:
        json.dump(sampled_decoded_sequences, f)
    with open(create_filename("sampled_target_logprobs", "json", output_dir), "w") as f:
        json.dump(sampled_target_logprobs, f)

    return sampled_sequences, sampled_decoded_sequences, sampled_target_logprobs

