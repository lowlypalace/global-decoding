import torch

from torch.nn.functional import log_softmax


def top_k_filtering(logits, top_k):
    # Retrieve the top_k logits and their indices for each sequence in the batch
    _, topk_indices = torch.topk(logits, top_k, dim=-1)
    # Create a mask of the same shape as logits, initialized to False
    mask = torch.ones_like(logits).scatter_(-1, topk_indices, 0).bool()
    # Set all elements of logits that are not in the top_k to -float("inf")
    logits[mask] = -float("inf")
    return logits


def get_logprobs(logits, index, pad_token_id, top_k=None):
    # Set logits for the pad_token_id to -float("Inf") to exclude them from top-k
    logits[:, :, pad_token_id] = -float("inf")

    # If top_k is specified, apply top-k filtering
    if top_k is not None:
        logits = top_k_filtering(logits, top_k)

    # Convert the (filtered) logits to log probabilities
    log_probs = log_softmax(logits, dim=-1)

    # Extract the log probabilities for the generated tokens
    selected_logprobs = torch.gather(log_probs, dim=-1, index=index).squeeze(-1)

    # Mask out the log probabilities for the pad_token_id
    pad_mask_selected = index.squeeze(-1) == pad_token_id
    selected_logprobs[pad_mask_selected] = 0

    return selected_logprobs


def create_index_tensor(sequences, input_ids):
    # Only use the IDs that were generated, excluding the input IDs
    gen_sequences = sequences[:, input_ids.shape[-1] :]
    # Add an additional dimension to the tensor to match the number of dimensions
    index = gen_sequences[:, :, None]
    return index


def get_logits(model, sequences):
    # Get the logits from the model
    return model(sequences, return_dict=True).logits


def sum_logprobs(logprobs):
    return torch.sum(logprobs, dim=-1)


def get_sequence_probs(model, sequences, top_k, pad_token_id, input_ids):
    sequences = torch.tensor(
        # [[50256, 464, 5940, 8123, 338, 50256],
        [
            [50256, 13, 198, 13, 50256, 50256],
        ]
    )
    with torch.no_grad():
        # Get the logits from the model
        logits = get_logits(model, sequences)
        # Get the index tensor for the generated tokens
        index = create_index_tensor(sequences, input_ids)
        # Get the log probabilities for the original sequence
        target_logprobs = get_logprobs(
            logits=logits, index=index, pad_token_id=pad_token_id
        )
        # Get the log probabilities for the proposed sequence
        proposal_logprobs = get_logprobs(
            logits=logits, index=index, pad_token_id=pad_token_id, top_k=top_k
        )

    # Sum the log probabilities for the entire sequence for both distributions
    target_logprob_sum = sum_logprobs(target_logprobs)
    proposal_logprob_sum = sum_logprobs(proposal_logprobs)

    return target_logprob_sum, proposal_logprob_sum

    # # Check if proposal_logprobs has any -inf values
    # # If so, print out the sequence that caused it
    # if torch.isinf(proposal_logprobs).any():
    #     print(sequences[torch.isinf(proposal_logprobs).any(dim=-1)])
