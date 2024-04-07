import json
import torch
import logging

from utils import (
    create_filename,
)


def generate_sequences(
    model,
    input_ids,
    max_length,
    top_k,
    batch_size,
    sequence_count,
    pad_token_id,
    eos_token_id,
    save_to_file,
    filename,
):
    # Calculate number of batches needed to generate the desired sequence_count
    num_batches = sequence_count // batch_size + (sequence_count % batch_size > 0)
    logging.info(
        f"Generating {sequence_count} sequences in {num_batches} batches of size {batch_size}..."
    )

    # Container for all generated sequences
    all_generated_sequences = []

    for _ in range(num_batches):
        # Generate a batch of sequences
        batch_sequences = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=batch_size,
        )

        # Collect the generated sequences
        all_generated_sequences.extend(batch_sequences)

        # If we've generated enough sequences, stop
        if len(all_generated_sequences) >= sequence_count:
            break

    # If we have more sequences than needed due to the last batch, truncate the list
    all_generated_sequences = all_generated_sequences[:sequence_count]

    if save_to_file:
        # Save the generated sequences to a file
        with open(create_filename(filename, "json", "sequences"), "w") as f:
            json.dump([g.tolist() for g in all_generated_sequences], f)

    logging.info(f"Generated {len(all_generated_sequences)} sequences in total.")

    return torch.stack(all_generated_sequences)


def load_preloaded_sequences(filename):
    with open(filename, "r") as f:
        preloaded_sequences = json.load(f)
    # Convert back to tensors
    preloaded_sequences = [
        torch.tensor(g, dtype=torch.long) for g in preloaded_sequences
    ]
    return preloaded_sequences
