import unittest
import torch

from src.sequences.generate_sequences_and_probs_hf import (
    generate_sequences_and_probs_hf,
)

from src.sequences.generate_sequences import generate_sequences

from src.sequences.sequences_and_probs import setup_model_and_tokenizer

from src.sequences.sequences_probs import get_sequences_probs

from src.utils.utils import (
    set_seed,
)


def setup(model_name="pythia-70m", precision="fp64", seed=42):
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set the seed for reproducibility
    set_seed(seed)
    # Initialize tokenizer and model, set to evaluation mode
    model, tokenizer = setup_model_and_tokenizer(model_name, precision, device)
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
