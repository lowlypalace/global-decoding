import torch
import logging
import json
from torch.nn.functional import log_softmax

def top_k_filtering(logits, top_k):
    top_k = min(top_k, logits.size(-1))  # Ensure top_k does not exceed number of logits
    # Compute the k-th best logit
    kth_best_logit = torch.topk(logits, top_k)[0][..., -1, None]
    # Create a mask for all logits that are less than the k-th best logit
    indices_to_remove = logits < kth_best_logit
    # Apply filtering
    logits_filtered = logits.masked_fill(indices_to_remove, -float("inf"))
    return logits_filtered

# def top_k_filtering(logits, top_k):
#     # Retrieve the top_k logits and their indices for each sequence in the batch
#     topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
#     # Create a mask of the same shape as logits, initialized to False
#     mask = torch.ones_like(logits).scatter_(-1, topk_indices, 0).bool()
#     # Set all elements of logits that are not in the top_k to -float("inf")
#     logits[mask] = -float("inf")

#     return logits


def mask_out_pad_token(log_probs, index, pad_token_id):
    # Create a mask that marks all pad_token_ids as True
    pad_mask = index.squeeze(-1) == pad_token_id
    # Find the first pad_token_id occurrence
    first_pad_mask = torch.cumsum(pad_mask, dim=1) == 1
    # Use the mask to set all but the first pad_token_id log_probs to 0
    # As the first pad_token_id is the end of the sequence, we do not want to mask it out
    log_probs[pad_mask & ~first_pad_mask] = 0
    return log_probs


# def get_logprobs(logits, index, pad_token_id, top_k=None, top_p=None):
def get_logprobs(logits, index, pad_token_id, top_k=None):
    # If top_k is specified, apply top-k filtering
    if top_k is not None:
        logits = top_k_filtering(logits, top_k)

    # Convert the (filtered) logits to log probabilities
    log_probs = log_softmax(logits, dim=-1)
    # Extract the log probabilities for the generated tokens
    selected_logprobs = torch.gather(log_probs, dim=-1, index=index).squeeze(-1)
    # Mask out the log probabilities for the padding tokens
    selected_logprobs = mask_out_pad_token(selected_logprobs, index, pad_token_id)

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


def save_logprobs(logprobs, filename):
    # Convert tensors to lists for JSON serialization
    logprobs_list = logprobs.tolist()
    # Save the target log probabilities in JSON format
    with open(filename, "w") as f:
        json.dump(logprobs_list, f)


def get_sequences_probs(
    model,
    sequences_ids,
    top_k,
    pad_token_id,
    input_ids,
    batch_size,
):
    # Calculate the number of batches
    num_sequences = sequences_ids.size(0)
    num_batches = (num_sequences + batch_size - 1) // batch_size

    # Save the log probabilities for each token in the sequences
    proposal_logprobs_tokens = torch.tensor([], device=sequences_ids.device)
    target_logprobs_tokens = torch.tensor([], device=sequences_ids.device)

    # Placeholder for the log probability sums
    target_logprob_sums = torch.tensor([], device=sequences_ids.device)
    proposal_logprob_sums = torch.tensor([], device=sequences_ids.device)

    logging.info(
        f"Computing probabilities for {num_sequences} sequences in {num_batches} batches of size {batch_size}..."
    )
    with torch.no_grad():
        for i in range(num_batches):
            # Compute the start and end indices for the current batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_sequences)

            # Slice the sequences to obtain the current batch
            sequences_ids_batch = sequences_ids[start_idx:end_idx]

            # Get the logits from the model for the current batch
            logits = get_logits(model, sequences_ids_batch)

            # Get the index tensor for the generated tokens in the current batch
            index = create_index_tensor(sequences_ids_batch, input_ids)

            # Get the log probabilities for the original sequence in the current batch
            target_logprobs = get_logprobs(
                logits=logits, index=index, pad_token_id=pad_token_id
            )

            # Get the log probabilities for the proposed sequence in the current batch
            proposal_logprobs = get_logprobs(
                logits=logits,
                index=index,
                pad_token_id=pad_token_id,
                top_k=top_k,
            )

            proposal_logprobs_tokens = torch.cat(
                (proposal_logprobs_tokens, proposal_logprobs)
            )
            target_logprobs_tokens = torch.cat(
                (target_logprobs_tokens, target_logprobs)
            )

            # Sum the log probabilities for the entire sequence for both distributions
            target_logprob_sum = sum_logprobs(target_logprobs)
            proposal_logprob_sum = sum_logprobs(proposal_logprobs)

            # Check for non-finite values and log if found
            if not torch.isfinite(target_logprob_sum).all():
                logging.warning(
                    f"Non-finite values detected in target log probabilities for batch {i}."
                )

            if not torch.isfinite(proposal_logprob_sum).all():
                logging.warning(
                    f"Non-finite values detected in proposal log probabilities for batch {i}."
                )

            # Append the results to the placeholders
            target_logprob_sums = torch.cat((target_logprob_sums, target_logprob_sum))
            proposal_logprob_sums = torch.cat(
                (proposal_logprob_sums, proposal_logprob_sum)
            )

    return (
        target_logprob_sums,
        proposal_logprob_sums,
        proposal_logprobs_tokens,
        target_logprobs_tokens,
    )
