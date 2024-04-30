import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    GPTNeoXForCausalLM,
)

from .sequences_probs import get_sequences_probs
from .generate_sequences import generate_sequences

from utils import timer


def generate_sequences_and_probs(args, output_subdir):
    # Parse command-line arguments
    top_k = args.top_k
    sequence_count = args.sequence_count
    max_length = args.max_length
    text = args.text
    batch_size_seq = args.batch_size_seq
    batch_size_prob = args.batch_size_prob
    model_name = args.model_name
    seed = args.seed
    device = torch.device(args.device)

    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Load model and tokenizer based on the selected model
    if model_name.startswith("pythia"):
        tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}")
        model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{model_name}")
    else:  # Default to gpt2 or gpt2-large
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        # Convert the model to double precision to avoid floating point discrepancies
        model.double()

    # Set the model to evaluation mode
    model.eval()
    # Move the model to the specified device
    model.to(device)
    # Assume max_model_length is the maximum sequence length the model can handle
    max_model_length = model.config.max_position_embeddings
    # Set the padding token to the EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set the padding side to the right
    tokenizer.padding_side = "right"
    # Set the text to the EOS token if it is not set
    if text is None:
        text = tokenizer.eos_token

    # Encode the input text to tensor
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(
        device
    )
    # Calculate the max_length so it is bound by the model context length
    max_length = max_length if max_length is not None else max_model_length

    # Generate sequences
    with timer("Generating new sequences"):
        sequences, decoded_sequences = generate_sequences(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=max_length,
            top_k=top_k,
            sequence_count=sequence_count,
            batch_size=batch_size_seq,
            output_subdir=output_subdir,
        )

    # Get the probabilities for the generated sequences
    with timer("Computing probabilities"):
        target_logprobs, proposal_logprobs = get_sequences_probs(
            model=model,
            sequences=sequences,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            input_ids=input_ids,
            batch_size=batch_size_prob,
            output_subdir=output_subdir,
        )

    # target_logpropbs are probabilities sampled from the global unnormalized distribution
    # proposal_logprobs are probabilities sampled from the local normalized distribution

    return sequences, decoded_sequences, target_logprobs, proposal_logprobs