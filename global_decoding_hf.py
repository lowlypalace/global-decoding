import torch
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import (
    create_filename,
    generate_sequences,
    load_preloaded_sequences,
    top_k_batch_filtering,
)


def indicator_top_k(sequence):
    # In our case, we can simply return 1 as we are using top-k sampling
    return 1


def get_original_logprobs(logits, index):
    # Convert logits to log probabilities
    original_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
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
    proposal_distribution = torch.nn.functional.log_softmax(filtered_logits, dim=-1)
    # Extract the log probabilities for the generated tokens from the proposal distribution
    gathered_proposal_logprobs = torch.gather(
        proposal_distribution, dim=-1, index=index
    ).squeeze(-1)

    return gathered_proposal_logprobs


def create_index_tensor(generated_ids):
    # Create an index tensor that identifies the positions of the generated tokens
    index = generated_ids[:, 1:].unsqueeze(-1)
    return index


def get_logits(model, generated_ids):
    # Slice off the last token from each sequence and get the logits
    return model(generated_ids[:, :-1], return_dict=True).logits


def get_sequence_probs(model, generated_ids, top_k):
    with torch.no_grad():
        # Get the logits from the model
        logits = get_logits(model, generated_ids)
        # Get the index tensor for the generated tokens
        index = create_index_tensor(generated_ids)
        # Get the log probabilities for the original sequence
        gathered_original_logprobs = get_original_logprobs(logits, index)
        # Get the log probabilities for the proposed sequence
        gathered_proposal_logprobs = get_proposal_logprobs(logits, top_k, index)

    # Sum the log probabilities for the entire sequence for both distributions
    original_logprob_sum = torch.sum(gathered_original_logprobs).item()
    proposal_logprob_sum = torch.sum(gathered_proposal_logprobs).item()

    return original_logprob_sum, proposal_logprob_sum


def metropolis_hastings(
    tokenizer,
    model,
    sequence_count,
    top_k,
    burnin,
    sequences,
    device,
):
    # List to store the generated samples, each sample is a tuple of (sequence, prob_sequence, prob_proposal)
    samples = []

    # Calculate the number of burn-in samples
    burnin_index = int(burnin * sequence_count)

    # Get the probabilities for the current sequence
    current_sequence = sequences[0]

    global_logprob_current, local_logprob_current = get_sequence_probs(
        model=model,
        generated_ids=current_sequence,
        top_k=top_k,
        device=device,
    )

    # This is a top-level loop to generate multiple sequences
    for i in range(1, sequence_count):
        # Get the sequence to propose
        proposed_sequence = sequences[i]

        # Calculate the probabilities for the current and proposed sequences
        (
            global_logprob_proposed,
            local_logprob_proposed,
        ) = get_sequence_probs(
            model=model,
            generated_ids=proposed_sequence,
            top_k=top_k,
            device=device,
        )

        # Calculate the acceptance ratio
        numerator = (
            global_logprob_proposed
            + indicator_top_k(proposed_sequence)
            + local_logprob_current
        )
        denominator = (
            global_logprob_current
            + indicator_top_k(current_sequence)
            + local_logprob_proposed
        )
        log_acceptance_ratio = numerator - denominator

        # Accept or reject the new sequence based on the acceptance ratio
        if np.log(np.random.uniform(0, 1)) < log_acceptance_ratio:
            current_sequence = proposed_sequence
            global_logprob_current = global_logprob_proposed
            local_logprob_current = local_logprob_proposed

        # After burn-in period, add the current state to the list of samples
        if i >= burnin_index:
            # Decode the generated sequence
            decoded_seq = tokenizer.decode(current_sequence, skip_special_tokens=True)
            # Append the decoded sequence and its probabilities to the samples list
            samples.append((decoded_seq, global_logprob_current, local_logprob_current))

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

    # Preloaded sequences to use
    preload_sequences = False
    sequences_filename = "sequences/generated_sequences.json_03-04-2024_16-30-56.json"

    # Encode the input text to tensor
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(
        device
    )

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
        sequences = generate_sequences(
            tokenizer=tokenizer,
            model=model,
            input_ids=input_ids,
            max_length=max_length,
            max_model_length=max_model_length,
            top_k=top_k,
            save_to_file=True,
            num_return_sequences=sequence_count,
        )

    # TODO: compute the probabilities for the generated sequences in advance

    # Run the Independent Metropolis-Hastings algorithm
    generated_samples = metropolis_hastings(
        tokenizer=tokenizer,
        model=model,
        sequence_count=sequence_count,
        top_k=top_k,
        burnin=burnin,
        sequences=sequences,
        device=device,
    )

    # Extract the probabilities from the generated samples
    generated_probs = [sample[1] for sample in generated_samples]

    # Plot the distribution of the generated probabilities
    plot_mcmc_distribution(generated_probs, plot_type="histogram", show=False)
    plot_mcmc_distribution(generated_probs, plot_type="kde", show=False)


if __name__ == "__main__":
    main()
