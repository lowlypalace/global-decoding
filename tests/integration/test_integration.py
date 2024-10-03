import unittest
import torch

from src.sequences.generate_sequences_and_probs_hf import (
    generate_sequences_and_probs_hf,
)
from src.sequences.generate_sequences_util import generate_sequences_util
from src.sequences.sequences_and_probs import ModelHandler
from src.sequences.sequences_probs import get_sequences_probs

from src.utils.utils import (
    set_seed,
)

# Utility function to set up the model and tokenizer
def setup(model_name="pythia-70m", precision="fp64", seed=42):
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set the seed for reproducibility
    set_seed(seed)
    # Initialize the model handler
    model_handler = ModelHandler(model_name, precision, device)
    # Get model and tokenizer
    model, tokenizer = model_handler.get_model_and_tokenizer()
    # Encode the input text
    input_ids = tokenizer.encode(tokenizer.eos_token, return_tensors="pt").to(device)

    return tokenizer, model, input_ids

class TestIntegration(unittest.TestCase):
    def test_top_k(self):

        # Reset model and tokenizer
        tokenizer, model, input_ids = setup()

        # Get sequences and their probabilities using custom implementation
        sequences_ids_custom, sequences_decoded_custom = generate_sequences_util(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=100,
            top_k=100,
            top_p=None,
            sequence_count=100,
            batch_size=16,
        )
        (
            target_logprobs_custom,
            proposal_logprobs_custom,
            target_logprobs_tokens_custom,
            proposal_logprobs_tokens_custom,
            target_normalize_constants,
            proposal_normalize_constants,
            _,
            _
        ) = get_sequences_probs(
            model=model,
            sequences_ids=sequences_ids_custom,
            top_k=100,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            input_ids=input_ids,
            batch_size=16,
        )

        # Reset model and tokenizer
        tokenizer, model, input_ids = setup()

        # Get sequences and their probabilities using Hugging Face implementation
        (
            sequences_ids_hf,
            sequences_decoded_hf,
            target_logprobs_hf,
            proposal_logprobs_hf,
            target_logprobs_tokens_hf,
            proposal_logprobs_tokens_hf,
        ) = generate_sequences_and_probs_hf(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=100,
            top_k=100,
            top_p=None,
            sequence_count=100,
            batch_size=16,
        )

        self.assertEqual(sequences_decoded_custom, sequences_decoded_hf)
        self.assertTrue(torch.allclose(target_logprobs_custom, target_logprobs_hf, rtol=1e-03, atol=1e-03))
        self.assertTrue(torch.allclose(proposal_logprobs_custom, proposal_logprobs_hf, rtol=1e-03, atol=1e-03))
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
        # Reset model and tokenizer
        tokenizer, model, input_ids = setup()

        # Get sequences and their probabilities using custom implementation
        sequences_ids_custom, sequences_decoded_custom = generate_sequences_util(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=100,
            top_k=None,
            top_p=0.6,
            sequence_count=100,
            batch_size=16,
        )
        (
            target_logprobs_custom,
            proposal_logprobs_custom,
            target_logprobs_tokens_custom,
            proposal_logprobs_tokens_custom,
            target_normalize_constants,
            proposal_normalize_constants,
            _,
            _
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
            target_logprobs_tokens_hf,
            proposal_logprobs_tokens_hf,
        ) = generate_sequences_and_probs_hf(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=100,
            top_k=None,
            top_p=0.6,
            sequence_count=100,
            batch_size=16,
        )

        self.assertEqual(sequences_decoded_custom, sequences_decoded_hf)
        self.assertTrue(torch.allclose(target_logprobs_custom, target_logprobs_hf, rtol=1e-03, atol=1e-03))
        self.assertTrue(torch.allclose(proposal_logprobs_custom, proposal_logprobs_hf, rtol=1e-03, atol=1e-03))
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
        sequences_ids, sequences_decoded = generate_sequences_util(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=100,
            top_k=None,
            top_p=None,
            sequence_count=100,
            batch_size=16,
        )
        (
            target_logprobs,
            proposal_logprobs,
            target_logprobs_tokens,
            proposal_logprobs_tokens,
            target_normalize_constants,
            proposal_normalize_constants,
            _,
            _
        ) = get_sequences_probs(
            model=model,
            sequences_ids=sequences_ids,
            top_k=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            input_ids=input_ids,
            batch_size=16,
        )

        self.assertTrue(torch.allclose(target_logprobs, proposal_logprobs, rtol=1e-03, atol=1e-03))
        self.assertTrue(
            torch.allclose(
                proposal_logprobs_tokens,
                target_logprobs_tokens,
                rtol=1e-03,
                atol=1e-03,
            )
        )

    # TODO: Add tests for MCMC and evaluations
    # TODO: Add tests with different floating point precisions


if __name__ == "__main__":
    unittest.main()
