import torch
import numpy as np
import random

import plotly.graph_objects as go
import plotly.figure_factory as ff

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import (
    create_filename,
    calculate_context_length,
    predict_logits,
    top_k_filtering,
)


def indicator_top_k(sequence):
    # In our case, we can simply return 1 as we are using top-k sampling
    return 1


def generate_sequence(
    tokenizer, model, input_ids, max_length, top_k, max_model_length, device
):
    # Clone the initial input_ids tensor to avoid modifying the original
    curr_input_ids = input_ids.clone()
    # Initialize sequence length
    seq_length = 0
    # Calculate the max_length so it is bound by the model context length
    max_length = calculate_context_length(input_ids, max_length, max_model_length)

    # Store product of original probabilities given by the model
    original_probs = []
    # Store the product of the proposal probabilities
    proposal_probs = []

    # Loop to generate a single sequence
    while True:
        # Retrieve the logits for the last token from the output
        last_token_logits = predict_logits(curr_input_ids, model)
        # Calculate probabilities from logits before top-k filtering
        original_probs_distribution = torch.nn.functional.log_softmax(
            last_token_logits, dim=-1
        )
        # Get top-k values
        _, top_indices = torch.topk(last_token_logits, top_k)

        # Apply top-k filtering to logits
        filtered_logits = top_k_filtering(last_token_logits, top_indices)
        # Normalize the filtered logits to probabilities
        proposal_probs_distribution = torch.nn.functional.log_softmax(filtered_logits, dim=-1)
        # Sample from the filtered distribution
        next_token = torch.multinomial(proposal_probs_distribution.exp(), num_samples=1).to(device)

        # Append the probabilities of the chosen token
        original_prob = original_probs_distribution[0, next_token.item()].item()
        original_probs.append(original_prob)
        proposal_prob = proposal_probs_distribution[0, next_token.item()].item()
        proposal_probs.append(proposal_prob)

        # Check for end of sequence token
        if next_token.item() == tokenizer.eos_token_id:
            break

        # If a maximum length is set and we have reached it, break the loop
        if max_length is not None and seq_length >= max_length:
            break

        # Concatenate the sampled token to form the extended sequence
        curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)

        # Increment sequence length
        seq_length += 1

    # Decode the generated sequence
    decoded_seq = tokenizer.decode(curr_input_ids[0], skip_special_tokens=True)

    # Calculate the product of the probabilities
    prob_sequence = np.sum(original_probs)
    prob_proposal = np.sum(proposal_probs)

    return decoded_seq, prob_sequence, prob_proposal


def metropolis_hastings(
    tokenizer,
    model,
    text,
    sequence_count,
    max_length,
    max_model_length,
    top_k,
    burnin,
    device,
):
    # Encode the input text to tensor
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(
        device
    )

    # List to store the generated samples, each sample is a tuple of (sequence, prob_sequence, prob_proposal)
    samples = []

    # Calculate the number of burn-in samples
    burnin_index = int(burnin * sequence_count)

    # Sample initial sequence
    current_sequence, prob_current, prob_proposal_current = generate_sequence(
        tokenizer=tokenizer,
        model=model,
        input_ids=input_ids,
        max_length=max_length,
        top_k=top_k,
        max_model_length=max_model_length,
        device=device,
    )

    # This is a top-level loop to generate multiple sequences
    for i in range(sequence_count):

        # Generate a single sequence
        proposed_sequence, prob_proposed, prob_proposal_proposed = generate_sequence(
            tokenizer=tokenizer,
            model=model,
            input_ids=input_ids,
            max_length=max_length,
            top_k=top_k,
            max_model_length=max_model_length,
            device=device,
        )

        # Calculate the acceptance ratio
        numerator = (
            prob_proposed + indicator_top_k(proposed_sequence) + prob_proposal_current
        )
        denominator = (
            prob_current + indicator_top_k(current_sequence) + prob_proposal_proposed
        )
        acceptance_ratio = min(1, np.exp(numerator - denominator))

        # Accept or reject the new sequence based on the acceptance ratio
        if random.uniform(0, 1) < acceptance_ratio:
            current_sequence = proposed_sequence
            prob_current = prob_proposed
            prob_proposal_current = prob_proposal_proposed

        # After burn-in period, add the current state to the list of samples
        if i >= burnin_index:
            samples.append((current_sequence, prob_current, prob_proposal_current))

    return samples


def plot_mcmc_distribution(samples, plot_type="histogram", show=True):
    if plot_type == "histogram":
        # Create the histogram data
        trace = go.Histogram(
            x=samples,
            histnorm="probability",
            nbinsx=30,
        )
        layout = go.Layout(
            title="Probability Distribution of MCMC Samples",
            xaxis=dict(title="Sample Value"),
            yaxis=dict(title="Probability Density"),
            bargap=0.2,
        )
        fig = go.Figure(data=[trace], layout=layout)
    elif plot_type == "kde":
        # Generate a kernel density estimate
        # Using Plotly's figure factory to create the KDE plot
        fig = ff.create_distplot(
            [samples],
            group_labels=["KDE"],
            bin_size=0.2,
            show_hist=False,
            show_rug=False,
        )
        fig.update_layout(
            title="Kernel Density Estimate of MCMC Samples",
            xaxis=dict(title="Sample Value"),
            yaxis=dict(title="Density"),
        )
    else:
        raise ValueError("Invalid plot_type. Use 'histogram' or 'kde'.")
    # Write the plot to an HTML file
    fig.write_html(create_filename(f"mcmc_{plot_type}", "html"))
    # Plot the figure
    if show:
        fig.show()


def main():
    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pre-trained model tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
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
    sequence_count = 1000
    # Maximum length of a sequence
    # This can be set to None to disable the maximum length constraint
    max_length = 10
    # Burn-in period as a fraction of the total number of samples
    burnin = 0.2

    # Run the Independent Metropolis-Hastings algorithm
    generated_samples = metropolis_hastings(
        tokenizer=tokenizer,
        model=model,
        text=text,
        sequence_count=sequence_count,
        max_length=max_length,
        max_model_length=max_model_length,
        top_k=top_k,
        burnin=burnin,
        device=device,
    )

    # Extract the probabilities from the generated samples
    generated_probs = [sample[1] for sample in generated_samples]

    # Plot the distribution of the generated probabilities
    plot_mcmc_distribution(generated_probs, plot_type="histogram", show=False)
    plot_mcmc_distribution(generated_probs, plot_type="kde", show=False)


if __name__ == "__main__":
    main()
