import numpy as np


def indicator_top_k(sequence):
    # In our case, we can simply return 1 as we are using top-k sampling
    return 1


def metropolis_hastings(
    tokenizer, sequence_count, burnin, sequences, target_logprobs, proposal_logprobs
):
    # List to store the generated samples, each sample is a tuple of (sequence, prob_sequence, prob_proposal)
    samples = []

    # Calculate the number of burn-in samples
    burnin_index = int(burnin * sequence_count)

    # Get the first sequence and its probabilities
    current_sequence = sequences[0]
    logprob_target_current, logprob_proposal_current = (
        target_logprobs[0],
        proposal_logprobs[0],
    )

    # This is a top-level loop to generate multiple sequences
    for i in range(1, sequence_count):
        # Get the sequence to propose
        proposed_sequence = sequences[i]
        # Get the probabilities for the proposed sequences
        logprob_target_proposed, logprob_proposal_proposed = (
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

        # After burn-in period, add the current state to the list of samples
        if i >= burnin_index:
            # Decode the generated sequence
            decoded_seq = tokenizer.decode(current_sequence, skip_special_tokens=True)
            # Append the decoded sequence and its probabilities to the samples list
            samples.append(
                (decoded_seq, logprob_target_current, logprob_proposal_current)
            )

    return samples
