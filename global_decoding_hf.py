import torch
import numpy as np
import random
from datetime import datetime
import plotly.graph_objects as go
import plotly.figure_factory as ff

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import (
    create_filename,
)


def indicator_top_k(sequence):
    # In our case, we can simply return 1 as we are using top-k sampling
    return 1


def generate_sequence(
    tokenizer, model, input_ids, max_length, top_k, max_model_length, device
):
    # Calculate the max_length so it is bound by the model context length
    max_length = min(max_length, max_model_length - input_ids.size(1))

    # Generate a single sequence
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_k=top_k,
        do_sample=True,
        # num_return_sequences=1,
    )

    return generated_ids


def top_k_filtering(logits, top_k):
    # Retrieve the top_k logits and their indices
    _, topk_indices = torch.topk(logits, top_k, dim=-1)
    # Create a mask of the same shape as logits, initialized to True
    mask = torch.ones_like(logits, dtype=torch.bool)
    # Set the mask to False for the top_k indices
    mask.scatter_(1, topk_indices, False)
    # Set all elements of logits that are not in the top_k to -float("inf")
    logits[mask] = -float("inf")
    return logits


def consume_sequence(tokenizer, model, generated_ids, top_k, device):
    with torch.no_grad():
        #
        logits = model(generated_ids[:, :-1], return_dict=True).logits
        # Convert logits to log probabilitiesxs
        original_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Creates an index that identifies the positions of the generated tokens
        index = generated_ids[:, 1:].unsqueeze(-1)
        # Extract their log probabilities from the original_logprobs
        gathered_original_logprobs = torch.gather(
            original_logprobs, dim=-1, index=index
        ).squeeze(-1)

        # Initialize tensors to store the log probabilities
        sequence_original_logprobs = torch.zeros(
            generated_ids.size(1) - 1, device=device
        )
        sequence_proposal_logprobs = torch.zeros(
            generated_ids.size(1) - 1, device=device
        )

        for i in range(generated_ids.size(1) - 1):
            # Select the log probability of the token that was actually generated
            sequence_original_logprobs[i] = gathered_original_logprobs[:, i]

            # Apply top-k filtering to create the proposal distribution
            topk_logits = logits[:, i, :].clone()
            filtered_logits = top_k_filtering(topk_logits, top_k)
            proposal_distribution = torch.nn.functional.log_softmax(
                filtered_logits, dim=-1
            )

            # Select the log probability of the token that was actually generated from the proposal distribution
            sequence_proposal_logprobs[i] = torch.gather(
                proposal_distribution, 1, generated_ids[:, i + 1].unsqueeze(-1)
            ).squeeze(-1)

    # Decode the generated sequence
    decoded_seq = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Sum the log probabilities for the entire sequence for both distributions
    original_logprob_sum = torch.sum(sequence_original_logprobs).item()
    proposal_logprob_sum = torch.sum(sequence_proposal_logprobs).item()

    return decoded_seq, original_logprob_sum, proposal_logprob_sum


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
    generated_ids = generate_sequence(
        tokenizer=tokenizer,
        model=model,
        input_ids=input_ids,
        max_length=max_length,
        top_k=top_k,
        max_model_length=max_model_length,
        device=device,
    )

    current_sequence, prob_current, prob_proposal_current = consume_sequence(
        tokenizer=tokenizer,
        model=model,
        generated_ids=generated_ids,
        top_k=top_k,
        device=device,
    )

    # This is a top-level loop to generate multiple sequences
    for i in range(sequence_count):
        # Generate a single sequence
        generated_ids = generate_sequence(
            tokenizer=tokenizer,
            model=model,
            input_ids=input_ids,
            max_length=max_length,
            top_k=top_k,
            max_model_length=max_model_length,
            device=device,
        )

        # Calculate the probabilities for the current and proposed sequences
        proposed_sequence, prob_proposed, prob_proposal_proposed = consume_sequence(
            tokenizer=tokenizer,
            model=model,
            generated_ids=generated_ids,
            top_k=top_k,
            device=device,
        )

        # Calculate the acceptance ratio
        numerator = (
            prob_proposed + indicator_top_k(proposed_sequence) + prob_proposal_current
        )
        denominator = (
            prob_current + indicator_top_k(current_sequence) + prob_proposal_proposed
        )
        log_acceptance_ratio = numerator - denominator

        # Accept or reject the new sequence based on the acceptance ratio
        if np.log(random.uniform(0, 1)) < log_acceptance_ratio:
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
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side="left")
    tokenizer.padding_side = "left"

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
    # This can be set to None to disable the maximum length constraint
    max_length = 10
    # Burn-in period as a fraction of the total number of samples
    burnin = 0.2

    # Run the Independent Metropolis-Hastings algorithm
    start_time = datetime.now()
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
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Duration: {duration}")

    # Extract the probabilities from the generated samples
    generated_probs = [sample[1] for sample in generated_samples]

    # Plot the distribution of the generated probabilities
    plot_mcmc_distribution(generated_probs, plot_type="histogram", show=False)
    plot_mcmc_distribution(generated_probs, plot_type="kde", show=False)


if __name__ == "__main__":
    main()
