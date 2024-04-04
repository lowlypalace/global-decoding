import os
import torch
import json
from datetime import datetime

def top_k_batch_filtering(logits, top_k):
    # Retrieve the top_k logits and their indices for each sequence in the batch
    _, topk_indices = torch.topk(logits, top_k, dim=-1)
    # Create a mask of the same shape as logits, initialized to False
    mask = torch.ones_like(logits).scatter_(
        -1, topk_indices, 0
    ).bool()
    # Set all elements of logits that are not in the top_k to -float("inf")
    logits[mask] = -float("inf")
    return logits

def load_preloaded_sequences(filename):
    with open(filename, "r") as f:
        preloaded_sequences = json.load(f)
    # Convert back to tensors
    preloaded_sequences = [
        torch.tensor(g, dtype=torch.long) for g in preloaded_sequences
    ]
    return preloaded_sequences


def generate_sequences(
    tokenizer,
    model,
    input_ids,
    max_length,
    top_k,
    max_model_length,
    num_return_sequences,
    save_to_file=False,
    filename="generated_sequences.json",
):
    # Calculate the max_length so it is bound by the model context length
    max_length = min(max_length, max_model_length - input_ids.size(1))

    # Generate sequences
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_k=top_k,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )

    if save_to_file:
        # Save the generated sequences to a file
        with open(create_filename(filename, "json", "sequences"), "w") as f:
            json.dump([g.tolist() for g in generated_ids], f)

    return generated_ids


def top_k_filtering(logits, top_indices):
    # Create a mask of the same shape as logits, initialized to True
    mask = torch.ones_like(logits, dtype=torch.bool)
    # Set the mask to False for the top_k indices
    mask[0][top_indices] = False
    # Set all elements of logits that are not in the top_k to -inf
    logits[mask] = -float("inf")

    return logits


def predict_logits(curr_input_ids, model):
    with torch.no_grad():
        # We pass our input_ids to the model to get the output.
        outputs = model(curr_input_ids)
    # The output of the model is a tuple, where the first element contains the logits.
    predictions = outputs[0]
    # Retrieve the logits for the last token from the output
    last_token_logits = predictions[:, -1, :]

    return last_token_logits


def calculate_context_length(input_ids, max_length, max_model_length):
    # Get the length of the input_ids tensor
    input_length = input_ids.size(1)
    # Calculate the max_length based on the input length and model's max position embeddings
    max_length = (
        max_model_length - input_length
        if max_length is None
        else min(max_length, max_model_length - input_length)
    )
    return max_length


def create_filename(name, extension, directory="plots"):
    # Get the current time
    current_time = datetime.now()
    # Format the time in a user-friendly format
    time_str = current_time.strftime("%d-%m-%Y_%H-%M-%S")
    # Create the filename with the current time
    filename = f"{name}_{time_str}.{extension}"
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Define the full path for the file
    full_path = os.path.join(directory, filename)

    return full_path
