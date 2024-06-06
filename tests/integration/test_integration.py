import unittest
import torch
import os

from transformers import AutoTokenizer, GPTNeoXForCausalLM

from src.sequences.generate_sequences_and_probs_hf import (
    generate_sequences_and_probs_hf,
)

from src.sequences.generate_sequences import generate_sequences

from src.sequences.sequences_probs import get_sequences_probs

from src.utils.utils import (
    set_seed,
)

# TODO: convert this to a proper integration test


def setup():
    # Initialize tokenizer and model, set to evaluation mode
    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-70m")
    model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/pythia-70m")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set the model to evaluation mode
    model.eval()
    # Move the model to the specified device
    model.to(device)
    # Set the model precision
    model.double()

    # Set the padding token to the EOS token
    tokenizer.pad_token = tokenizer.eos_token
    # Set the padding side to the right
    tokenizer.padding_side = "right"

    # Set the seed for reproducibility
    set_seed(42)

    # Encode the input text
    input_ids = tokenizer.encode(tokenizer.eos_token, return_tensors="pt").to(device)

    return tokenizer, model, input_ids


class TestImplementations(unittest.TestCase):

    def test_top_k(self):

        tokenizer, model, input_ids = setup()

        # Get sequences and their probabilities using custom implementation
        sequences_ids_custom, sequences_decoded_custom = generate_sequences(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=10,
            top_k=100,
            top_p=None,
            sequence_count=10,
            batch_size=16,
        )
        (
            target_logprobs_custom,
            proposal_logprobs_custom,
            proposal_logprobs_tokens_custom,
            target_logprobs_tokens_custom,
        ) = get_sequences_probs(
            model=model,
            sequences_ids=sequences_ids_custom,
            top_k=100,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            input_ids=input_ids,
            batch_size=16,
        )

        # Reset model
        tokenizer, model, input_ids = setup()

        # Get sequences and their probabilities using Hugging Face implementation
        (
            sequences_ids_hf,
            sequences_decoded_hf,
            target_logprobs_hf,
            proposal_logprobs_hf,
            proposal_logprobs_tokens_hf,
            target_logprobs_tokens_hf,
        ) = generate_sequences_and_probs_hf(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=10,
            top_k=100,
            top_p=None,
            sequence_count=10,
            batch_size=16,
        )

        self.assertEqual(sequences_decoded_custom, sequences_decoded_hf)
        self.assertTrue(
            torch.allclose(
                target_logprobs_custom, target_logprobs_hf, rtol=1e-03, atol=1e-03
            )
        )
        self.assertTrue(
            torch.allclose(
                proposal_logprobs_custom, proposal_logprobs_hf, rtol=1e-03, atol=1e-03
            )
        )
        self.assertTrue(
            torch.allclose(
                proposal_logprobs_tokens_custom,
                proposal_logprobs_tokens_hf,
                rtol=1e-03,
                atol=1e-03,
            )
        )
        self.assertTrue(
            torch.allclose(
                target_logprobs_tokens_custom,
                target_logprobs_tokens_hf,
                rtol=1e-03,
                atol=1e-03,
            )
        )

    def test_top_p(self):

        tokenizer, model, input_ids = setup()

        # Get sequences and their probabilities using custom implementation
        sequences_ids_custom, sequences_decoded_custom = generate_sequences(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=10,
            top_k=None,
            top_p=0.6,
            sequence_count=10,
            batch_size=16,
        )
        (
            target_logprobs_custom,
            proposal_logprobs_custom,
            proposal_logprobs_tokens_custom,
            target_logprobs_tokens_custom,
        ) = get_sequences_probs(
            model=model,
            sequences_ids=sequences_ids_custom,
            top_k=None,
            top_p=0.6,
            pad_token_id=tokenizer.pad_token_id,
            input_ids=input_ids,
            batch_size=16,
        )

        # Reset model
        tokenizer, model, input_ids = setup()

        # Get sequences and their probabilities using Hugging Face implementation
        (
            sequences_ids_hf,
            sequences_decoded_hf,
            target_logprobs_hf,
            proposal_logprobs_hf,
            proposal_logprobs_tokens_hf,
            target_logprobs_tokens_hf,
        ) = generate_sequences_and_probs_hf(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=10,
            top_k=None,
            top_p=0.6,
            sequence_count=10,
            batch_size=16,
        )

        self.assertEqual(sequences_decoded_custom, sequences_decoded_hf)
        self.assertTrue(
            torch.allclose(
                target_logprobs_custom, target_logprobs_hf, rtol=1e-03, atol=1e-03
            )
        )
        self.assertTrue(
            torch.allclose(
                proposal_logprobs_custom, proposal_logprobs_hf, rtol=1e-03, atol=1e-03
            )
        )
        self.assertTrue(
            torch.allclose(
                proposal_logprobs_tokens_custom,
                proposal_logprobs_tokens_hf,
                rtol=1e-03,
                atol=1e-03,
            )
        )
        self.assertTrue(
            torch.allclose(
                target_logprobs_tokens_custom,
                target_logprobs_tokens_hf,
                rtol=1e-03,
                atol=1e-03,
            )
        )

    def test_no_top_k_no_top_p(self):

        tokenizer, model, input_ids = setup()

        # Get sequences and their probabilities using custom implementation
        sequences_ids, sequences_decoded = generate_sequences(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=10,
            top_k=None,
            top_p=None,
            sequence_count=10,
            batch_size=16,
        )
        (
            target_logprobs,
            proposal_logprobs,
            proposal_logprobs_tokens,
            target_logprobs_tokens,
        ) = get_sequences_probs(
            model=model,
            sequences_ids=sequences_ids,
            top_k=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            input_ids=input_ids,
            batch_size=16,
        )

        self.assertTrue(
            torch.allclose(target_logprobs, proposal_logprobs, rtol=1e-03, atol=1e-03)
        )
        self.assertTrue(
            torch.allclose(
                proposal_logprobs_tokens,
                target_logprobs_tokens,
                rtol=1e-03,
                atol=1e-03,
            )
        )


if __name__ == "__main__":
    unittest.main()
