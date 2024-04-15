import numpy as np


def indicator_top_k(sequence):
    # In our case, we can simply return 1 as we are using top-k sampling
    return 1


def metropolis_hastings(
    tokenizer, sequence_count, burnin, sequences, target_logprobs, proposal_logprobs, rate
):
    # List to store the generated samples
    sampled_sequences = []
    sampled_target_logprobs = []

    # Calculate the number of burn-in samples
    burnin_index = int(burnin * sequence_count)

    # Get the first sequence and its probabilities
    current_sequence = sequences[0]
    logprob_target_current, logprob_proposal_current = (
        target_logprobs[0].item(),
        proposal_logprobs[0].item(),
    )

    # This is a top-level loop to generate multiple sequences
    for i in range(1, sequence_count):
        # Get the sequence to propose
        proposed_sequence = sequences[i]
        # Get the probabilities for the proposed sequences
        logprob_target_proposed, logprob_proposal_proposed = (
            target_logprobs[i].item(),
            proposal_logprobs[i].item(),
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
            # Decode the generated sequence
            decoded_seq = tokenizer.decode(current_sequence, skip_special_tokens=True)
            # Append the decoded sequence and its probabilities to samples
            sampled_sequences.append(decoded_seq)
            sampled_target_logprobs.append(logprob_target_current)

    return sampled_sequences, sampled_target_logprobs
