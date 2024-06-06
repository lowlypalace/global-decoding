import os
import logging

import torch

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    GPTNeoXForCausalLM,
)

from src.utils.utils import timer, save_to_json, load_from_json

from src.sequences.sequences_probs import get_sequences_probs
from src.sequences.generate_sequences import generate_sequences

from src.mcmc.plots import plot_distribution


def setup_model_and_tokenizer(model_name, precision, device):
    # Load model and tokenizer based on the selected model
    if model_name.startswith("pythia"):
        tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}")
        model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{model_name}")
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set the model precision
    if precision == "fp16":
        model = model.half()
    elif precision == "fp64":
        model = model.double()

    # Set the model to evaluation mode
    model.eval()
    # Move the model to the specified device
    model.to(device)

    # Set the padding token to the EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token_id = tokenizer.eos_token_id
    # Set the padding side to the right
    tokenizer.padding_side = "right"

    return model, tokenizer


def load_sequences(output_subdir, device):
    sequences_ids = load_from_json(os.path.join(output_subdir, "sequences_ids"))
    sequences_decoded = load_from_json(os.path.join(output_subdir, "sequences_decoded"))
    # Convert the list of lists to a list of tensors
    return torch.tensor(sequences_ids).to(device), sequences_decoded


def save_sequences(output_subdir, sequences_ids, sequences_decoded):
    # Save the encoded and decoded sequences
    save_to_json(
        # Convert the list of tensors to a list of lists
        [sequence_ids.tolist() for sequence_ids in sequences_ids],
        "sequences_ids",
        output_subdir,
    )
    save_to_json(sequences_decoded, "sequences_decoded", output_subdir)

def set_max_length(model, max_length):
    # Assume max_model_length is the maximum sequence length the model can handle
    max_model_length = model.config.max_position_embeddings
    # Calculate the max_length so it is bound by the model context length
    max_length = max_length if max_length is not None else max_model_length

    return max_length

def load_probs(output_subdir):
    target_logprobs = load_from_json(os.path.join(output_subdir, "logprobs_target"))
    proposal_logprobs = load_from_json(os.path.join(output_subdir, "logprobs_proposal"))
    target_logprobs_tokens = load_from_json(
        os.path.join(output_subdir, "logprobs_target_tokens")
    )
    proposal_logprobs_tokens = load_from_json(
        os.path.join(output_subdir, "logprobs_proposal_tokens")
    )
    return (
        target_logprobs,
        proposal_logprobs,
        target_logprobs_tokens,
        proposal_logprobs_tokens,
    )


def probs_exist(output_subdir):
    return os.path.exists(
        os.path.join(output_subdir, "logprobs_target.json")
    ) and os.path.exists(os.path.join(output_subdir, "logprobs_proposal.json"))


def save_probs(
    output_subdir,
    target_logprobs,
    proposal_logprobs,
    target_logprobs_tokens,
    proposal_logprobs_tokens,
):
    save_to_json(target_logprobs, "logprobs_target", output_subdir)
    save_to_json(proposal_logprobs, "logprobs_proposal", output_subdir)
    save_to_json(target_logprobs_tokens, "logprobs_target_tokens", output_subdir)
    save_to_json(proposal_logprobs_tokens, "logprobs_proposal_tokens", output_subdir)


def generate_sequences_and_probs(args, output_subdir):
    # Parse command-line arguments
    top_k = args.top_k
    top_p = args.top_p
    sequence_count = args.sequence_count
    max_length = args.max_length
    text = args.text
    batch_size_seq = args.batch_size_seq
    batch_size_prob = args.batch_size_prob
    model_name = args.model_name
    precision = args.precision
    preload_sequences = args.preload_sequences
    device = torch.device(args.device)

    # Setup the model and tokenizer
    logging.info("Setting up the model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_name, precision, device)

    # Set the text to the EOS token if it is not set
    if text is None:
        text = tokenizer.eos_token

    # Encode the input text to tensor
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(
        device
    )

    max_length = set_max_length(model, max_length)

    if preload_sequences:
        logging.info("Loading preloaded sequences...")
        sequences_ids, sequences_decoded = load_sequences(output_subdir, device)
    else:
        with timer("Generating new sequences"):
            sequences_ids, sequences_decoded = generate_sequences(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_length=max_length,
                top_k=top_k,
                top_p=top_p,
                sequence_count=sequence_count,
                batch_size=batch_size_seq,
            )

        # Convert tensors to lists
        logging.info("Saving the generated sequences...")
        save_sequences(output_subdir, sequences_ids, sequences_decoded)

    # Get the probabilities for the generated sequences
    if preload_sequences and probs_exist(output_subdir):
        logging.info("Loading precomputed probabilities...")
        (
            target_logprobs,
            proposal_logprobs,
            target_logprobs_tokens,
            proposal_logprobs_tokens,
        ) = load_probs(output_subdir)

    else:
        with timer("Computing probabilities"):
            (
                target_logprobs,
                proposal_logprobs,
                proposal_logprobs_tokens,
                target_logprobs_tokens,
            ) = get_sequences_probs(
                model=model,
                sequences_ids=sequences_ids,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                input_ids=input_ids,
                batch_size=batch_size_prob,
            )

        logging.info("Saving the log probabilities...")
        save_probs(
            output_subdir,
            target_logprobs,
            proposal_logprobs,
            target_logprobs_tokens,
            proposal_logprobs_tokens,
        )

    logging.info("Plotting the log probabilities distributions...")
    # Plot the distribution of the target log-probabilities
    plot_distribution(
        target_logprobs,
        plot_type="histogram",
        prefix="target_logprobs",
        show=False,
        output_dir=os.path.join(output_subdir, "plots"),
    )
    # Plot the distribution of the proposal log-probabilities
    plot_distribution(
        proposal_logprobs,
        plot_type="histogram",
        prefix="proposal_logprobs",
        show=False,
        output_dir=os.path.join(output_subdir, "plots"),
    )

    # Convert the list of tensors to a list of lists
    sequences_ids = [sequence_ids.tolist() for sequence_ids in sequences_ids]

    return (
        sequences_ids,
        sequences_decoded,
        target_logprobs,
        proposal_logprobs,
    )
