import json
import torch
import logging
import argparse
import logging
import torch
import os
import time
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from utils import (
    create_filename,
)

from sequence_probability import get_sequence_probs
from utils import setup_logging, save_args, get_timestamp


# Set the environment variable for memory allocation strategy
# TODO: Check if this is needed
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def generate_sequences(
    model,
    tokenizer,
    input_ids,
    max_length,
    top_k,
    batch_size,
    sequence_count,
    output_dir
):
    # Calculate number of batches needed to generate the desired sequence_count
    num_batches = sequence_count // batch_size + (sequence_count % batch_size > 0)
    logging.info(
        f"Generating {sequence_count} sequences in {num_batches} batches of size {batch_size}..."
    )

    # Container for all generated sequences
    all_generated_sequences = []

    for _ in range(num_batches):
        # Generate a batch of sequences
        batch_sequences = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=batch_size,
        )

        # Collect the generated sequences
        all_generated_sequences.extend(batch_sequences)

        # If we've generated enough sequences, stop
        if len(all_generated_sequences) >= sequence_count:
            break

    # If we have more sequences than needed due to the last batch, truncate the list
    all_generated_sequences = all_generated_sequences[:sequence_count]

    logging.info(f"Generated {len(all_generated_sequences)} sequences in total.")

    # Decode sequences to text
    decoded_sequences = [tokenizer.decode(g, skip_special_tokens=True) for g in all_generated_sequences]

    # Save the encoded sequences
    logging.info("Saving the generated sequences...")
    with open(create_filename("sequences_ids", "json", output_dir), "w") as f:
        json.dump([g.tolist() for g in all_generated_sequences], f)

    # Save the decoded sequences
    with open(create_filename("sequences_decoded", "json", output_dir), "w") as f:
        json.dump(decoded_sequences, f)


    return torch.stack(all_generated_sequences), decoded_sequences


def load_preloaded_sequences(filename):
    with open(filename, "r") as f:
        preloaded_sequences = json.load(f)
    # Convert back to tensors
    preloaded_sequences = [
        torch.tensor(g, dtype=torch.long) for g in preloaded_sequences
    ]
    return preloaded_sequences



# Define the function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate text sequences.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "pythia"],
        help="Model to use for text generation. Supports GPT-2 and Pythia.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to use as a prompt. Defaults to the EOS token.",
    )
    parser.add_argument(
        "--top_k", type=int, default=100, help="Top-k value for text generation."
    )
    parser.add_argument(
        "--sequence_count",
        type=int,
        default=100,
        help="Number of sequence samples to generate and use for MCMC analysis.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length. If not provided, it will be set to the maximum model length minus the length of the input text.",
    )
    parser.add_argument(
        "--preload_sequences",
        action="store_true",
        help="Use preloaded sequences instead of generating new ones.",
    )
    parser.add_argument(
        "--sequences_filename",
        type=str,
        default="generated_sequences",
        help="Filename for preloaded sequences.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to use for computation. Defaults to "cuda" if available.',
    )
    parser.add_argument(
        "--batch_size_seq",
        type=int,
        default=64,
        help="Batch size for generating sequences.",
    )
    parser.add_argument(
        "--batch_size_prob",
        type=int,
        default=16,
        help="Batch size for computing probabilities.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("output", "seq"),
        help="Directory to save the output files.",
    )

    args = parser.parse_args()
    return args


def main():
    # Parse command-line arguments
    args = parse_args()
    top_k = args.top_k
    sequence_count = args.sequence_count
    max_length = args.max_length
    preload_sequences = args.preload_sequences
    sequences_filename = args.sequences_filename
    text = args.text
    batch_size_seq = args.batch_size_seq
    batch_size_prob = args.batch_size_prob
    model_name = args.model_name
    seed = args.seed
    output_dir = args.output_dir
    device = torch.device(args.device)

    # Add a directory with a timestamp to the output directory
    output_dir = os.path.join(output_dir, get_timestamp())
    # Create a directory to save the output files
    os.makedirs(output_dir, exist_ok=True)
    # Save log messages to a file
    setup_logging(log_file=os.path.join(output_dir, "log.txt"))
    # Save command-line arguments to JSON
    save_args(args, output_dir)

    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Load model and tokenizer based on the selected model
    if args.model_name == "pythia":
        tokenizer = AutoTokenizer.from_pretrained("facebook/pythia")
        model = AutoModelForCausalLM.from_pretrained("facebook/pythia")
    else:  # Default to gpt2 or gpt2-large
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set the padding side to the left
    tokenizer.padding_side = "left"
    # Set the model to evaluation mode
    model.eval()
    # Move the model to the specified device
    model.to(device)
    # Convert the model to double precision to avoid floating point discrepancies
    model.double()
    # Assume max_model_length is the maximum sequence length the model can handle
    max_model_length = model.config.max_position_embeddings
    # Set the padding token to the EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set the text to the EOS token if it is not set
    if text is None:
        text = tokenizer.eos_token

    # Generate sequences and save them to a file
    if preload_sequences:
        logging.info("Loading preloaded sequences...")
        sequences = load_preloaded_sequences(sequences_filename)
        if len(sequences) != sequence_count:
            raise ValueError(
                f"Number of sequences in the file ({len(sequences)}) does not match sequence_count ({sequence_count})."
            )
    else:
        # Encode the input text to tensor
        input_ids = tokenizer.encode(
            text, add_special_tokens=True, return_tensors="pt"
        ).to(device)
        # Calculate the max_length so it is bound by the model context length
        max_length = max_length if max_length is not None else max_model_length
        # Generate sequences
        start_time = time.time()
        logging.info("Generating new sequences...")
        # TODO: get logits from generate method
        sequences, decoded_sequences = generate_sequences(
            model=model,
            tokenizer = tokenizer,
            input_ids=input_ids,
            max_length=max_length,
            top_k=top_k,
            sequence_count=sequence_count,
            batch_size=batch_size_seq,
            output_dir=os.path.join(output_dir),
        )
        end_time = time.time()
        logging.info(
            f"Generated {sequence_count} sequences in {end_time - start_time:.2f} seconds."
        )

    # Get the probabilities for the generated sequences
    start_time = time.time()
    logging.info("Computing probabilities for the generated sequences...")
    # target_logpropbs are probabilities sampled from the global unnormalized distribution
    # proposal_logprobs are probabilities sampled from the local normalized distribution
    target_logprobs, proposal_logprobs = get_sequence_probs(
        model=model,
        sequences=sequences,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
        input_ids=input_ids,
        batch_size=batch_size_prob,
        output_dir=os.path.join(output_dir),
    )
    end_time = time.time()
    logging.info(f"Computed probabilities in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
