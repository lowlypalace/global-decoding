import torch
import numpy as np
import csv
import re
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import plotly.graph_objects as go


def normalization_constant(logits, top_indices):
    # Normalize probabilities of all of the logits (not filtered)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # Get only top-k probabilities
    top_k_probs = probs[0][top_indices]
    # Calculate the inverse sum of the top-k (unnormalized) probabilities
    normalization_const = 1.0 / torch.sum(top_k_probs, dim=-1)

    return normalization_const.item()


def top_k_filtering(logits, top_indices):
    # Create a mask of the same shape as logits, initialized to True
    mask = torch.ones_like(logits, dtype=torch.bool)
    # Set the mask to False for the top_k indices
    mask[0][top_indices] = False
    # Set all elements of logits that are not in the top_k to -inf
    logits[mask] = -float("inf")

    return logits


def predict_logits(curr_input_ids, model):
    with torch.no_grad():
        # We pass our input_ids to the model to get the output.
        outputs = model(curr_input_ids)
    # The output of the model is a tuple, where the first element contains the logits.
    predictions = outputs[0]
    # Retrieve the logits for the last token from the output
    last_token_logits = predictions[:, -1, :]

    return last_token_logits


def calculate_context_length(input_ids, max_length, max_model_length):
    # Get the length of the input_ids tensor
    input_length = input_ids.size(1)
    # Calculate the max_length based on the input length and model's max position embeddings
    max_length = (
        max_model_length - input_length
        if max_length is None
        else min(max_length, max_model_length - input_length)
    )
    return max_length


def generate_sequence(
    tokenizer, model, input_ids, max_length, top_k, max_model_length, device
):
    # Initialize list to store local constants for the sequence
    constants_list = []
    # Clone the initial input_ids tensor to avoid modifying the original
    curr_input_ids = input_ids.clone()
    # Initialize sequence length
    seq_length = 0
    # Calculate the max_length so it is bound by the model context length
    max_length = calculate_context_length(input_ids, max_length, max_model_length)

    # Loop to generate a single sequence
    while True:
        # Retrieve the logits for the last token from the output
        last_token_logits = predict_logits(curr_input_ids, model)
        # Get top-k values
        _, top_indices = torch.topk(last_token_logits, top_k)

        # Calculate local constant
        constants_list.append(normalization_constant(last_token_logits, top_indices))

        # Apply top-k filtering to logits
        filtered_logits = top_k_filtering(last_token_logits, top_indices)
        # Normalize the filtered logits to probabilities
        probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
        # Sample from the filtered distribution
        next_token = torch.multinomial(probs, num_samples=1).to(device)

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

    # Calculate the product of constants for the sequence
    constants_product = np.prod(constants_list)
    # Decode the generated sequence
    decoded_seq = tokenizer.decode(curr_input_ids[0], skip_special_tokens=True)

    return constants_product, constants_list, seq_length, decoded_seq


# Main function to generate text and compute local decoding constants
def generate_and_compute_constants(
    tokenizer,
    model,
    text,
    top_k_values,
    sequence_count,
    max_length,
    max_model_length,
    device,
    verbose=False,
):
    # Encode the input text to tensor
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(
        device
    )
    # Define dictionary that will hold the product of local constants for each sequence
    # The sequences are mapped to the top k value
    constants_products = {top_k: [] for top_k in top_k_values}
    # Store local constants for each sequence
    constants_lists = {top_k: [] for top_k in top_k_values}
    # Store sequence lengths
    sequence_lengths = {top_k: [] for top_k in top_k_values}
    # Store the decoded sequences
    decoded_sequences = {top_k: [] for top_k in top_k_values}

    # Iterate over provided top k values
    for top_k in top_k_values:
        print()
        print(f"Generating sequences for top k = {top_k}")

        # This is a top-level loop to generate multiple sequences
        for i in range(sequence_count):
            if i % 10 == 0:
                print(f"Generating sequence {i} of {sequence_count}")

            # Generate a single sequence
            product, constants_list, seq_length, decoded_seq = generate_sequence(
                tokenizer=tokenizer,
                model=model,
                input_ids=input_ids,
                max_length=max_length,
                top_k=top_k,
                max_model_length=max_model_length,
                device=device,
            )

            # Store the product of constants and sequence length for each sequence
            constants_products[top_k].append(product)
            constants_lists[top_k].append(constants_list)
            sequence_lengths[top_k].append(seq_length)
            decoded_sequences[top_k].append(decoded_seq)

            # Print for debugging
            if verbose:
                print(f"Constants product: {product}")
                print(f"Constants list: {constants_list}")
                print(f"Sequence length: {seq_length}")
                print(f"Generated sequence: {decoded_seq}")

    return constants_products, constants_lists, sequence_lengths, decoded_sequences


def create_filename(name, extension):
    # Get the current time
    current_time = datetime.now()
    # Format the time in a user-friendly format
    time_str = current_time.strftime("%d-%m-%Y_%H-%M-%S")
    # Create the filename with the current time
    filename = f"{name}_{time_str}.{extension}"

    return filename


# Function to plot histograms of constants using Plotly
def plot_histograms(constants_products, decoded_sequences, show=True):
    data = []

    # Loop through the dictionary to create a histogram for each set of constants
    for top_k, constants in constants_products.items():
        # Convert constants to log scale
        log_constants = np.log(constants)
        # Generate hover text
        hover_text = [
            "c_alpha: {:.2f}<br>Sequence: {}".format(
                const, "<br>".join(re.findall(".{1,90}(?:\\s|$)", seq))
            )
            for const, seq in zip(constants, decoded_sequences[top_k])
        ]
        # Create a histogram for the current set of constants
        histogram = go.Histogram(
            x=log_constants,
            nbinsx=30,
            name=f"Top K = {top_k}",
            opacity=0.5,
            hoverinfo="all",
            hovertext=hover_text,
        )
        data.append(histogram)

    # Create a layout for the plot
    layout = go.Layout(
        title="Histogram of Decoding Constants",
        xaxis=dict(title="Decoding Constant ln(c_alpha)"),
        yaxis=dict(title="Frequency"),
        barmode="overlay",
    )

    # Create a figure with the data and layout
    fig = go.Figure(data=data, layout=layout)
    # Save graph
    fig.write_html(create_filename("histogram", "html"))
    # Show the figure
    if show:
        fig.show()


# Plotting the constants against their respective sequence lengths
def plot_constants_vs_length(
    constants_products, sequence_lengths, decoded_sequences, show=True
):
    data = []

    for top_k in constants_products:
        # Generate hover text
        hover_text = [
            "c_alpha: {:.2f}<br>Length: {}<br>Sequence: {}".format(
                const, length, "<br>".join(re.findall(".{1,90}(?:\\s|$)", seq))
            )
            for const, length, seq in zip(
                constants_products[top_k],
                sequence_lengths[top_k],
                decoded_sequences[top_k],
            )
        ]

        # Convert constants to log scale
        log_constants = np.log(constants_products[top_k])
        # Create a scatter plot for the current set of constants
        scatter = go.Scatter(
            x=sequence_lengths[top_k],
            y=log_constants,
            mode="markers",
            name=f"Top K = {top_k}",
            hoverinfo="all",
            text=hover_text,
        )
        data.append(scatter)

    # Create a layout for the plot
    layout = go.Layout(
        title="Decoding Constants vs. Sequence Length",
        xaxis=dict(title="Sequence Length"),
        yaxis=dict(title="Decoding Constant ln(c_alpha)"),
        hovermode="closest",
    )
    # Create a figure with the data and layout
    fig = go.Figure(data=data, layout=layout)
    # Save graph
    fig.write_html(create_filename("scatterplot", "html"))
    # Show the figure
    if show:
        fig.show()


# Save the data to a CSV file
def save_data(
    constants_products, constants_lists, sequence_lengths, decoded_sequences, filename
):
    # Open the file in write mode
    with open(filename, mode="w", newline="") as file:
        # Create a CSV writer object
        csv_writer = csv.writer(file)
        # Write the header
        csv_writer.writerow(
            [
                "top_k",
                "constants_product",
                "constants_list",
                "sequence_length",
                "decoded_sequence",
            ]
        )
        # Write data rows
        for top_k in constants_products:
            for constant_product, constant_list, seq_length, decoded_seq in zip(
                constants_products[top_k],
                constants_lists[top_k],
                sequence_lengths[top_k],
                decoded_sequences[top_k],
            ):
                csv_writer.writerow(
                    [top_k, constant_product, constant_list, seq_length, decoded_seq]
                )


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
    # Top-k values to use
    top_k_values = [5, 10, 50, 100, 500, 1000, 5000, 10000]
    # Number of sequences to generate for each top-k setting
    sequence_count = 100
    # Maximum length of a sequence
    # This can be set to None to disable the maximum length constraint
    max_length = None

    # Generate sequences and compute constants
    constants_products, constants_lists, sequence_lengths, decoded_sequences = (
        generate_and_compute_constants(
            tokenizer=tokenizer,
            model=model,
            text=text,
            top_k_values=top_k_values,
            sequence_count=sequence_count,
            max_length=max_length,
            max_model_length=max_model_length,
            device=device,
            verbose=False,
        )
    )

    # Each bar represents the number of sequences that resulted in a specifc range of `c_alpha` values
    # Separate color is used for each top-k setting
    plot_histograms(
        constants_products=constants_products,
        decoded_sequences=decoded_sequences,
        show=False,
    )
    # Each point represents a sequence
    # The x-coordinate representing the sequence length
    # The y-coordinate representing the `c_alpha` value
    plot_constants_vs_length(
        constants_products=constants_products,
        decoded_sequences=decoded_sequences,
        sequence_lengths=sequence_lengths,
        show=False,
    )
    # Save the data to a CSV file
    save_data(
        constants_products=constants_products,
        constants_lists=constants_lists,
        sequence_lengths=sequence_lengths,
        decoded_sequences=decoded_sequences,
        filename=create_filename("output", "csv"),
    )


if __name__ == "__main__":
    main()
