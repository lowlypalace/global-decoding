import torch

from torch.nn.functional import log_softmax

from utils import (
    top_k_batch_filtering,
)


def get_original_logprobs(logits, index):
    # Convert logits to log probabilities
    original_logprobs = log_softmax(logits, dim=-1)
    # Extract their log probabilities from the original log probabilities
    gathered_original_logprobs = torch.gather(
        original_logprobs, dim=-1, index=index
    ).squeeze(-1)

    return gathered_original_logprobs


def get_proposal_logprobs(logits, top_k, index):
    # Clone the logits to avoid modifying the original tensor
    filtered_logits = logits.clone()
    # Filter the logits using top-k filtering
    filtered_logits = top_k_batch_filtering(filtered_logits, top_k)
    # Convert the filtered logits to log probabilities
    proposal_distribution = log_softmax(filtered_logits, dim=-1)
    # Extract the log probabilities for the generated tokens from the proposal distribution
    gathered_proposal_logprobs = torch.gather(
        proposal_distribution, dim=-1, index=index
    ).squeeze(-1)

    return gathered_proposal_logprobs


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
        gathered_original_logprobs = get_original_logprobs(logits, index)
        # Get the log probabilities for the proposed sequence
        gathered_proposal_logprobs = get_proposal_logprobs(logits, top_k, index)

    # Sum the log probabilities for the entire sequence for both distributions
    original_logprob_sum = torch.sum(gathered_original_logprobs, dim=-1)
    proposal_logprob_sum = torch.sum(gathered_proposal_logprobs, dim=-1)

    return original_logprob_sum, proposal_logprob_sum
