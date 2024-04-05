import json
import torch

from utils import (
    create_filename,
)


def generate_sequences(
    tokenizer,
    model,
    input_ids,
    max_length,
    top_k,
    num_return_sequences,
    save_to_file=False,
    filename="generated_sequences.json",
    output_scores=False,
):
    # Generate sequences
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_k=top_k,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        output_scores=output_scores,
    )

    if save_to_file:
        # Save the generated sequences to a file
        with open(create_filename(filename, "json", "sequences"), "w") as f:
            json.dump([g.tolist() for g in generated_ids], f)

    return generated_ids


def load_preloaded_sequences(filename):
    with open(filename, "r") as f:
        preloaded_sequences = json.load(f)
    # Convert back to tensors
    preloaded_sequences = [
        torch.tensor(g, dtype=torch.long) for g in preloaded_sequences
    ]
    return preloaded_sequences
