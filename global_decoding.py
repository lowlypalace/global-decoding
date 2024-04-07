import argparse
import logging
import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set the environment variable for memory allocation strategy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from generate_sequences import (
    generate_sequences,
    load_preloaded_sequences,
)

from sequence_probability import get_sequence_probs
from metropolis_hastings import metropolis_hastings
from plots import plot_mcmc_distribution, plot_chain


# Set up logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# Define the function to parse command-line arguments
def parse_args(tokenizer):
    parser = argparse.ArgumentParser(
        description="Generate text sequences with GPT-2 and perform MCMC analysis."
    )

    parser.add_argument(
        "--text",
        type=str,
        default=tokenizer.eos_token,
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
        "--burnin",
        type=float,
        default=0.2,
        help="Burn-in period as a fraction of the total number of samples.",
    )
    parser.add_argument(
        "--preload_sequences",
        action="store_true",
        help="Use preloaded sequences instead of generating new ones.",
    )
    parser.add_argument(
        "--sequences_filename",
        type=str,
        default="sequences/generated_sequences.json",
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
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for generating sequences.",
    )

    args = parser.parse_args()
    return args


def main():
    setup_logging()

    # Load pre-trained model tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Set the padding side to the left
    tokenizer.padding_side = "left"
    # Load pre-trained model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # Set the model to evaluation mode
    model.eval()
    # Assume max_model_length is the maximum sequence length the model can handle
    max_model_length = model.config.max_position_embeddings
    # Set the padding token to the EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Parse command-line arguments
    args = parse_args(tokenizer)
    top_k = args.top_k
    sequence_count = args.sequence_count
    max_length = args.max_length
    burnin = args.burnin
    preload_sequences = args.preload_sequences
    sequences_filename = args.sequences_filename
    text = args.text
    batch_size = args.batch_size
    device = torch.device(args.device)

    # Move the model to the specified device
    model.to(device)

    # Generate sequences and save them to a file
    if preload_sequences:
        logging.info("Loading preloaded sequences...")
        sequences = load_preloaded_sequences(sequences_filename)
        if len(sequences) != sequence_count:
            raise ValueError(
                f"Number of sequences in the file ({len(sequences)}) does not match sequence_count ({sequence_count})."
            )
    else:
        logging.info("Generating new sequences...")
        # Encode the input text to tensor
        input_ids = tokenizer.encode(
            text, add_special_tokens=True, return_tensors="pt"
        ).to(device)
        # Calculate the max_length so it is bound by the model context length
        # max_length incudes both the input text and the generated text
        max_length = max_length if max_length is not None else max_model_length
        # Generate sequences
        sequences = generate_sequences(
            model=model,
            input_ids=input_ids,
            max_length=max_length,
            top_k=top_k,
            save_to_file=True,
            sequence_count=sequence_count,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            batch_size=batch_size,
            filename="generated_sequences",
        )

    logging.info("Computing probabilities for the generated sequences...")
    # Get the probabilities for the generated sequences
    global_logprobs, local_logprobs = get_sequence_probs(
        model=model,
        sequences=sequences,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
        input_ids=input_ids,
    )

    # Run the Independent Metropolis-Hastings algorithm
    logging.info("Running Independent Metropolis-Hastings algorithm...")
    sampled_sequences, sampled_probs = metropolis_hastings(
        tokenizer=tokenizer,
        sequence_count=sequence_count,
        burnin=burnin,
        sequences=sequences,
        target_logprobs=global_logprobs,
        proposal_logprobs=local_logprobs,
    )

    logging.info("Plotting the results...")
    # # Move the sampled probabilities to the CPU for plotting
    # sampled_probs = [s.cpu().numpy() for s in sampled_probs]
    # Plot the distribution of the generated probabilities
    plot_mcmc_distribution(sampled_probs, plot_type="histogram", show=False)
    plot_mcmc_distribution(sampled_probs, plot_type="kde", show=False)
    # Plot the chain of generated samples
    plot_chain(sampled_probs, burnin=burnin, show=False)


if __name__ == "__main__":
    main()
