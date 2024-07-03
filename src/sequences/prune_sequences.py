import logging


def contains_only_nonprintable(text):
    # Check if all characters in the text are non-printable
    return all(not char.isprintable() for char in text.strip())


def is_negative_inf(prop_prob):
    if prop_prob == float("-inf"):
        return True


def prune_sequences(
    args, sequences_ids, sequences_decoded, target_logprobs, proposal_logprobs
):
    logging.info(f"Pruning sequences...")
    # Filter out the sequences that contain only non-printable characters
    # These sequences raise divide by zero error in BLEU computation
    # https://github.com/huggingface/evaluate/issues/601
    filtered_data = [
        (seq_id, text, target_prob, prop_prob)
        for seq_id, text, target_prob, prop_prob in zip(
            sequences_ids, sequences_decoded, target_logprobs, proposal_logprobs
        )
        if not contains_only_nonprintable(text)
    ]

    logging.info(
        f"Removed {len(sequences_ids) - len(filtered_data)} sequences with non-printable characters"
    )
    # Filter out the sequences if the proposed proposal log probability is -inf
    # This is happening because of precision issues when getting logits from the model directly
    # https://github.com/huggingface/transformers/issues/31127
    filtered_data_ = [
        (seq_id, text, target_prob, prop_prob)
        for seq_id, text, target_prob, prop_prob in filtered_data
        if not is_negative_inf(prop_prob)
    ]

    logging.warning(
        f"Removed {len(filtered_data) - len(filtered_data_)} sequences with -inf proposal log probability"
    )

    # Unzip the filtered data back into separate lists
    sequences_ids, sequences_decoded, target_logprobs, proposal_logprobs = zip(
        *filtered_data_
    )

    # If we have more sequences than needed due to the last batch, truncate the lists
    sequences_ids = sequences_ids[: args.sequence_count]
    sequences_decoded = sequences_decoded[: args.sequence_count]
    target_logprobs = target_logprobs[: args.sequence_count]
    proposal_logprobs = proposal_logprobs[: args.sequence_count]

    logging.info(f"Truncated sequences to {len(sequences_ids)} number of sequences")

    return (
        list(sequences_ids),
        list(sequences_decoded),
        list(target_logprobs),
        list(proposal_logprobs),
    )
