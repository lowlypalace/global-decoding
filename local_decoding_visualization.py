import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import plotly.graph_objects as go

def normalization_constant(logits, top_indices):
    # Normalize probabilities of all of the logits (not filtered)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get only top-k probabilities
    top_k_probs = probs[0][top_indices]

    # Calculate the inverse sum of the top-k (unnormalized) probabilities
    normalization_const = 1 / torch.sum(top_k_probs, dim=-1)

    return normalization_const.item()

def top_k_filtering(logits, top_indices):
    # Create a mask of the same shape as logits, initialized to True
    mask = torch.ones_like(logits, dtype=torch.bool)

    # Set the mask to False for the top_k indices
    mask[0][top_indices] = False

    # Set all elements of logits that are not in the top_k to -inf
    logits[mask] = -float('inf')

    return logits

def predict_logits(curr_input_ids):
    # Check if the current input exceeds the maximum model length
    if curr_input_ids.size(1) >= max_model_length:

        # If it does, truncate the input to the maximum model length
        curr_input_ids = curr_input_ids[:, -max_model_length:]

    # We pass our input_ids to the model to get the output.
    outputs = model(curr_input_ids)

    # The output of the model is a tuple, where the first element contains the logits (raw, unnormalized scores for each possible next token).
    predictions = outputs[0]

    # Retrieve the logits for the last token from the output
    last_token_logits = predictions[:, -1, :]

    return last_token_logits

# Main function to generate text and compute local decoding constants
def generate_and_compute_constants(tokenizer, text, top_k_values, sequence_count):
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt').to(device)

    # Define dictionary that will hold the product of local constants for each sequence and map them to the top k value
    constants = {top_k: [] for top_k in top_k_values}

    # Store sequence lengths
    sequence_lengths = {top_k: [] for top_k in top_k_values}

    # Iterate over different top k values
    for top_k in top_k_values:
        print(f"Generating sequences for top k = {top_k}")

        # Now go over whatever number of sequence we need to generate
        for _ in range(sequence_count):

            local_constants = []
            curr_input_ids = input_ids.clone()
            sequence_length = 0  # Initialize sequence length

            # This is a loop to generate a single sequence
            while True:

                # Retrieve the logits for the last token from the output
                last_token_logits = predict_logits(curr_input_ids)

                # Get top-k values
                _, top_indices = torch.topk(last_token_logits, top_k)

                # Calculate local constant
                local_const = normalization_constant(last_token_logits, top_indices)
                local_constants.append(local_const)

                # Apply top-k filtering to logits (inplace)
                filtered_logits = top_k_filtering(last_token_logits, top_indices)

                # Normalize the filtered logits to probabilities
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)

                # Sample from the filtered distribution
                next_token = torch.multinomial(probs, num_samples=1).to(device)

                # We concatenate the sampled next_token to the original input_ids to form the extended sequence.
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)

                # Increment sequence length
                sequence_length += 1

                if next_token.item() == tokenizer.eos_token_id:
                    break

            # # Calculate c_alpha for the sequence
            c_alpha = np.prod(local_constants)
            constants[top_k].append(c_alpha)
            # Append the sequence length
            sequence_lengths[top_k].append(sequence_length)  # Append the sequence length

            # Print sequences for debugging
            # generated_text = tokenizer.decode(curr_input_ids[0], skip_special_tokens=True)
            # print(generated_text)

    return constants, sequence_lengths

# Function to plot histograms of constants using Plotly
def plot_histograms(constants_dict):
    # Create an empty list to hold the histogram data
    data = []

    # Loop through the constants_dict to create a histogram for each set of constants
    for top_k, constants in constants_dict.items():
        # Create a histogram for the current set of constants
        histogram = go.Histogram(
            x=constants,
            nbinsx=30,
            name=f"Top K = {top_k}",
            opacity=0.5
        )
        # Append the histogram to our data list
        data.append(histogram)

    # Create a layout for the plot
    layout = go.Layout(
        title='Histogram of Local Decoding Constants',
        xaxis=dict(title='Local Decoding Constant c_alpha'),
        yaxis=dict(title='Frequency'),
        barmode='overlay'
    )

    # Create a figure with the data and layout
    fig = go.Figure(data=data, layout=layout)

    # Save graph
    fig.write_html("histogram.html")

    # Show the figure
    fig.show()

"""Each bar represents the number of sequences that resulted in a particular range of `c_alpha` values, with a separate color for each top-k setting"""

# Plotting the constants against their respective sequence lengths
def plot_constants_vs_length(constants_dict, lengths_dict):
    data = []

    for top_k in constants_dict:
        scatter = go.Scatter(
            x=lengths_dict[top_k],
            y=constants_dict[top_k],
            mode='markers',
            name=f"Top K = {top_k}"
        )
        data.append(scatter)

    layout = go.Layout(
        title='Local Decoding Constants vs. Sequence Length',
        xaxis=dict(title='Sequence Length'),
        yaxis=dict(title='Local Decoding Constant c_alpha'),
        hovermode='closest'
    )

    fig = go.Figure(data=data, layout=layout)

    fig.write_html("scatter.html")

    fig.show()

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# Assume max_model_length is the maximum sequence length the model can handle
max_model_length = model.config.max_position_embeddings

# Define text, top_k_values, sequence_count, and max_length
text = "Hi, this is"
top_k_values = [5, 10, 50, 100, 500, 1000]
sequence_count = 100

# Generate sequences and compute constants
constants, sequence_lengths = generate_and_compute_constants(tokenizer, text, top_k_values, sequence_count)

plot_histograms(constants)
plot_constants_vs_length(constants, sequence_lengths)
