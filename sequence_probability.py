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


def get_target_logprobs(logits, index):
    # Convert logits to log probabilities
    target_distribution = log_softmax(logits, dim=-1)
    # Extract their log probabilities from the original log probabilities
    selected_target_logprobs = torch.gather(
        target_distribution, dim=-1, index=index
    ).squeeze(-1)

    return selected_target_logprobs


def get_proposal_logprobs(logits, top_k, index):
    # Clone the logits to avoid modifying the original tensor
    filtered_logits = logits.clone()
    # Filter the logits using top-k filtering
    filtered_logits = top_k_filtering(filtered_logits, top_k)
    # Convert the filtered logits to log probabilities
    proposal_distribution = log_softmax(filtered_logits, dim=-1)
    # Extract the log probabilities for the generated tokens from the proposal distribution
    selected_proposal_logprobs = torch.gather(
        proposal_distribution, dim=-1, index=index
    ).squeeze(-1)

    return selected_proposal_logprobs


def create_index_tensor(sequences):
    # Create an index tensor that identifies the positions of the generated tokens
    index = sequences[:, 1:].unsqueeze(-1)
    return index


def get_logits(model, sequences):
    # Slice off the last token from each sequence and get the logits
    return model(sequences[:, :-1], return_dict=True).logits


def get_sequence_probs(model, sequences, top_k):
    with torch.no_grad():
        # Get the logits from the model
        logits = get_logits(model, sequences)
        # Get the index tensor for the generated tokens
        index = create_index_tensor(sequences)
        # Get the log probabilities for the original sequence
        target_logprobs = get_target_logprobs(logits, index)
        # Get the log probabilities for the proposed sequence
        proposal_logprobs = get_proposal_logprobs(logits, top_k, index)

    # Sum the log probabilities for the entire sequence for both distributions
    target_logprob_sum = torch.sum(target_logprobs, dim=-1)
    proposal_logprob_sum = torch.sum(proposal_logprobs, dim=-1)

    return target_logprob_sum, proposal_logprob_sum
