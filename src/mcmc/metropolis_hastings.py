import numpy as np


def indicator_top_k(sequence):
    # In our case, we can simply return 1 as we are using top-k sampling
    return 1


def metropolis_hastings(
    sequence_count,
    sequences_ids,
    sequences_decoded,
    target_logprobs,
    proposal_logprobs,
    seed,
):

    # Reset the seed at the start of the function to ensure reproducibility
    np.random.seed(seed)

    # List to store the generated samples
    collected_sequences_ids = []
    collected_sequences_decoded = []
    collected_target_logprobs = []
    # Lists to store the deltas for the acceptance ratio (for plotting)
    logprob_diff_proposed = []
    logprob_diff_current = []
    # List to store the indices where the sequence changes (for plotting)
    sequence_change_indices = []

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
            target_logprobs[i],
            proposal_logprobs[i],
        )

        logprob_diff_proposed.append(
            logprob_target_proposed - logprob_proposal_proposed
        )
        logprob_diff_current.append(logprob_target_current - logprob_proposal_current)

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
            # Record the iteration index where the sequence changes
            sequence_change_indices.append(i)

        # After the burn-in period, add the current state to the list of samples at the specified rate
        collected_sequences_ids.append(current_sequence)
        # Append the decoded sequence and its probabilities to samples
        collected_sequences_decoded.append(current_decoded_seq)
        collected_target_logprobs.append(logprob_target_current)

    # Increment the seed after each run to ensure variability
    seed += 1

    return (
        collected_sequences_ids,
        collected_sequences_decoded,
        collected_target_logprobs,
        logprob_diff_proposed,
        logprob_diff_current,
        sequence_change_indices,
    )
