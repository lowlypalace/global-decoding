import logging

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
):

    # List to store the generated samples
    collected_sequences_ids = []
    collected_sequences_decoded = []
    collected_target_logprobs = []
    # Lists to store the deltas for the acceptance ratio for plotting
    logprob_diff_proposed = []
    logprob_diff_current = []
    # List to store the indices where the sequence changes for plotting
    sequence_change_indices = []

    # Get the first sequence and its probabilities
    current_sequence = sequences_ids[0]
    current_decoded_seq = sequences_decoded[0]
    current_target_logprob = target_logprobs[0]
    current_proposal_logprob = proposal_logprobs[0]

    # Collect the initial sequence
    collected_sequences_ids.append(current_sequence)
    collected_sequences_decoded.append(current_decoded_seq)
    collected_target_logprobs.append(current_target_logprob)

    # This is a top-level loop to generate multiple sequences
    for i in range(1, sequence_count):
        # Get the sequence to propose
        proposed_sequence = sequences_ids[i]
        # Get the probabilities for the proposed sequences
        proposed_target_logprob = target_logprobs[i]
        proposed_proposal_logprob = proposal_logprobs[i]

        # Skip the iteration if the proposed proposal log probability is -inf
        # This is happening because of precision issues when getting logits from the model directly
        # https://github.com/huggingface/transformers/issues/31127
        if proposed_proposal_logprob == float("-inf"):
            logprob_diff_proposed.append(
                None
            )  # Placeholder indicating skipped iteration
            logprob_diff_current.append(
                None
            )  # Placeholder indicating skipped iteration
            logging.warning(
                f"Skipping iteration {i} due to -inf proposal log probability."
            )
            continue

        # Calculate differences for plotting
        logprob_diff_proposed.append(
            proposed_target_logprob - proposed_proposal_logprob
        )
        logprob_diff_current.append(current_target_logprob - current_proposal_logprob)

        # Calculate the acceptance ratio
        numerator = (
            proposed_target_logprob
            + indicator_top_k(proposed_sequence)
            + current_proposal_logprob
        )
        denominator = (
            current_target_logprob
            + indicator_top_k(current_sequence)
            + proposed_proposal_logprob
        )
        log_acceptance_ratio = numerator - denominator

        # Accept or reject the new sequence based on the acceptance ratio
        if np.log(np.random.uniform(0, 1)) < log_acceptance_ratio:
            current_sequence = proposed_sequence
            current_decoded_seq = sequences_decoded[i]
            current_target_logprob = proposed_target_logprob
            current_proposal_logprob = proposed_proposal_logprob
            # Record the iteration index where the sequence changes
            sequence_change_indices.append(i - 1)

        # Append the current sequence and its probability to samples
        collected_sequences_ids.append(current_sequence)
        collected_sequences_decoded.append(current_decoded_seq)
        collected_target_logprobs.append(current_target_logprob)

    return (
        collected_sequences_ids,
        collected_sequences_decoded,
        collected_target_logprobs,
        logprob_diff_proposed,
        logprob_diff_current,
        sequence_change_indices,
    )
