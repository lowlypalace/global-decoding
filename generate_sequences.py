import torch
import numpy as np
import random
from datetime import datetime
import plotly.graph_objects as go
import plotly.figure_factory as ff

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import (
    generate_sequences,
)


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
    # Maximum length of a sequence
    # This can be set to None to disable the maximum length constraint
    max_length = 10
    # Number of sequences to generate
    num_return_sequences = 10

    # Encode the input text to tensor
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(
        device
    )

    # Generate sequences and save them to a file
    generate_sequences(
        tokenizer=tokenizer,
        model=model,
        input_ids=input_ids,
        max_length=max_length,
        max_model_length=max_model_length,
        top_k=top_k,
        save_to_file=True,
        num_return_sequences=num_return_sequences,
    )


if __name__ == "__main__":
    main()
