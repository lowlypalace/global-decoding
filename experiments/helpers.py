import torch


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
