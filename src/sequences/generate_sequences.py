import json
import torch
import logging
import logging
import torch

from utils import create_filename


def generate_sequences(
    model,
    tokenizer,
    input_ids,
    max_length,
    top_k,
    batch_size,
    sequence_count,
    output_subdir,
):
    # Calculate number of batches needed to generate the desired sequence_count
    num_batches = sequence_count // batch_size + (sequence_count % batch_size > 0)
    logging.info(
        f"Generating {sequence_count} sequences in {num_batches} batches of size {batch_size}..."
    )

    # Container for all generated sequences
    sequences_ids = []

    with torch.no_grad():
        for _ in range(num_batches):
            # Generate a batch of sequences
            sequences_ids_batch = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_k=top_k,
                do_sample=True,
                num_return_sequences=batch_size,
            )

            # Pad sequences in the batch to max_length
            padded_sequences_ids_batch = tokenizer.pad(
                {"input_ids": sequences_ids_batch},
                padding="max_length",  # Pads to a maximum length specified by the max_length parameter
                max_length=max_length,  # Define the total maximum length
                return_tensors="pt",
            ).to(input_ids.device)

            # Collect the generated sequences
            sequences_ids.extend(padded_sequences_ids_batch["input_ids"])

            # If we've generated enough sequences, stop
            if len(sequences_ids) >= sequence_count:
                break

    # If we have more sequences than needed due to the last batch, truncate the list
    sequences_ids = sequences_ids[:sequence_count]

    logging.info(f"Generated {len(sequences_ids)} sequences in total.")

    # Decode sequences to text
    sequences_decoded = tokenizer.batch_decode(sequences_ids, skip_special_tokens=True)

    # Convert the sequences to lists
    sequences_ids = [sequence_ids.tolist() for sequence_ids in sequences_ids]
    # Save the encoded sequences
    logging.info("Saving the generated sequences...")
    with open(create_filename("sequences_ids", "json", output_subdir), "w") as f:
        json.dump([g.tolist() for g in sequences_ids], f)

    # Save the decoded sequences
    with open(create_filename("sequences_decoded", "json", output_subdir), "w") as f:
        json.dump(sequences_decoded, f)

    return torch.stack(sequences_ids), sequences_decoded
