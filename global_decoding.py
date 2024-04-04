import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from generate_sequences import (
    generate_sequences,
    load_preloaded_sequences,
)

from sequence_probability import get_sequence_probs
from metropolis_hastings import metropolis_hastings
from plots import plot_mcmc_distribution


def main():
    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pre-trained model tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side="left")
    # Load pre-trained model
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Set the model to evaluation mode
    model.eval()
    # Assume max_model_length is the maximum sequence length the model can handle
    max_model_length = model.config.max_position_embeddings

    # Text to use as a prompt
    text = tokenizer.eos_token
    # Top-k value to use
    top_k = 100
    # Number of samples to generate
    sequence_count = 100
    # Maximum length of a sequence
    # If not provided, it will be set to the maximum model length minus the length of the input text
    max_length = 10
    # Burn-in period as a fraction of the total number of samples
    burnin = 0.2

    # Preloaded sequences to use
    preload_sequences = False
    sequences_filename = "sequences/generated_sequences.json_03-04-2024_16-30-56.json"

    # Generate sequences and save them to a file
    if preload_sequences:
        print("Loading preloaded sequences...")
        sequences = load_preloaded_sequences(sequences_filename)
        if len(sequences) != sequence_count:
            raise ValueError(
                f"Number of sequences in the file ({len(sequences)}) does not match sequence_count ({sequence_count})."
            )
    else:
        print("Generating new sequences...")
        # Encode the input text to tensor
        input_ids = tokenizer.encode(
            text, add_special_tokens=True, return_tensors="pt"
        ).to(device)
        # Calculate the max_length so it is bound by the model context length
        max_length = (
            max_length
            if max_length is not None
            else max_model_length - input_ids.size(1)
        )
        # Generate sequences
        sequences = generate_sequences(
            tokenizer=tokenizer,
            model=model,
            input_ids=input_ids,
            max_length=max_length,
            top_k=top_k,
            save_to_file=True,
            num_return_sequences=sequence_count,
        )

    print("Computing probabilities for the generated sequences...")
    # Get the probabilities for the generated sequences
    global_logprobs, local_logprobs = get_sequence_probs(
        model=model,
        sequences=sequences,
        top_k=top_k,
    )

    # Run the Independent Metropolis-Hastings algorithm
    print("Running Independent Metropolis-Hastings algorithm...")
    generated_samples = metropolis_hastings(
        tokenizer=tokenizer,
        sequence_count=sequence_count,
        burnin=burnin,
        sequences=sequences,
        target_logprobs=global_logprobs,
        proposal_logprobs=local_logprobs,
    )

    # Extract the probabilities from the generated samples
    generated_probs = [sample[1] for sample in generated_samples]

    print(len(generated_probs))

    # Plot the distribution of the generated probabilities
    plot_mcmc_distribution(generated_probs, plot_type="histogram", show=False)
    plot_mcmc_distribution(generated_probs, plot_type="kde", show=False)


if __name__ == "__main__":
    main()
