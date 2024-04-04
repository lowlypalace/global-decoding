import torch
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff


from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import (
    create_filename,
    generate_sequences,
    load_preloaded_sequences,
)

from sequence_probability import get_sequence_probs


def indicator_top_k(sequence):
    # In our case, we can simply return 1 as we are using top-k sampling
    return 1


def metropolis_hastings(
    tokenizer, sequence_count, burnin, sequences, target_logprobs, proposal_logprobs
):
    # List to store the generated samples, each sample is a tuple of (sequence, prob_sequence, prob_proposal)
    samples = []

    # Calculate the number of burn-in samples
    burnin_index = int(burnin * sequence_count)

    # Get the first sequence and its probabilities
    current_sequence = sequences[0]
    logprob_target_current, logprob_proposal_current = (
        target_logprobs[0],
        proposal_logprobs[0],
    )

    # This is a top-level loop to generate multiple sequences
    for i in range(1, sequence_count):
        # Get the sequence to propose
        proposed_sequence = sequences[i]
        # Get the probabilities for the proposed sequences
        logprob_target_proposed, logprob_proposal_proposed = (
            target_logprobs[i],
            proposal_logprobs[i],
        )

        # Calculate the acceptance ratio
        numerator = (
            logprob_target_proposed
            + indicator_top_k(proposed_sequence)
            + logprob_proposal_current
        )
        denominator = (
            logprob_target_current
            + indicator_top_k(current_sequence)
            + logprob_proposal_proposed
        )
        log_acceptance_ratio = numerator - denominator

        # Accept or reject the new sequence based on the acceptance ratio
        if np.log(np.random.uniform(0, 1)) < log_acceptance_ratio:
            current_sequence = proposed_sequence
            logprob_target_current = logprob_target_proposed
            logprob_proposal_current = logprob_proposal_proposed

        # After burn-in period, add the current state to the list of samples
        if i >= burnin_index:
            # Decode the generated sequence
            decoded_seq = tokenizer.decode(current_sequence, skip_special_tokens=True)
            # Append the decoded sequence and its probabilities to the samples list
            samples.append(
                (decoded_seq, logprob_target_current, logprob_proposal_current)
            )

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

    # Plot the distribution of the generated probabilities
    plot_mcmc_distribution(generated_probs, plot_type="histogram", show=False)
    plot_mcmc_distribution(generated_probs, plot_type="kde", show=False)


if __name__ == "__main__":
    main()
