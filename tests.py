import unittest
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from sequence_probability import (
    top_k_filtering,
    create_index_tensor,
    sum_logprobs,
    mask_out_pad_token,
    get_logprobs,
    get_logits,
)


class TestTopKFiltering(unittest.TestCase):
    def test_single_value(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        top_k = 1
        filtered_logits = top_k_filtering(logits, top_k)
        self.assertTrue(
            torch.allclose(
                filtered_logits, torch.tensor([[-float("inf"), -float("inf"), 3.0]])
            )
        )

    def test_all_values(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        top_k = 3
        filtered_logits = top_k_filtering(logits, top_k)
        self.assertTrue(torch.allclose(filtered_logits, logits))

    def test_batched_logits(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        top_k = 2
        filtered_logits = top_k_filtering(logits, top_k)
        expected_output = torch.tensor(
            [[-float("inf"), 2.0, 3.0], [-float("inf"), 5.0, 6.0]]
        )
        self.assertTrue(torch.allclose(filtered_logits, expected_output))

    def test_top_k_filtering_on_large_batch(self):
        # Testing on a larger batch to ensure the function scales
        batch_size = 64
        sequence_length = 10
        vocab_size = 100
        top_k = 5

        logits = torch.randn(batch_size, sequence_length, vocab_size)
        filtered_logits = top_k_filtering(logits, top_k)

        # For each item in the batch, check that there are exactly top_k values not equal to -inf
        for i in range(batch_size):
            for j in range(sequence_length):
                self.assertEqual(
                    torch.isfinite(filtered_logits[i, j]).sum().item(), top_k
                )


class TestSequenceProbability(unittest.TestCase):
    def test_with_pad_tokens(self):
        input_ids = torch.tensor([[50256]])
        sequences = torch.tensor(
            [
                [50256, 464, 5940, 8123, 338, 4452, 50256],
                [50256, 960, 383, 281, 1468, 12, 50256],
                [50256, 49, 1523, 284, 46085, 262, 50256],
                [50256, 13, 198, 13, 50256, 50256, 50256],
            ]
        )
        expected = torch.tensor(
            [
                [[464], [5940], [8123], [338], [4452], [50256]],
                [[960], [383], [281], [1468], [12], [50256]],
                [[49], [1523], [284], [46085], [262], [50256]],
                [[13], [198], [13], [50256], [50256], [50256]],
            ]
        )

        output = create_index_tensor(sequences, input_ids)
        torch.testing.assert_close(output, expected)

    def test_sum_logprobs(self):
        # Create a tensor of log probabilities for testing
        logprobs = torch.tensor(
            [[-2.3026, -1.6094, -1.2039], [-1.6094, -2.3026, -0.9163]]
        )
        # Call the sum_logprobs method
        summed_logprobs = sum_logprobs(logprobs)
        # The expected result is the sum along the last dimension (dim=-1)
        expected_sum = torch.tensor([-5.1159, -4.8283])
        # Assert that the summed log probabilities match the expected values
        torch.testing.assert_close(summed_logprobs, expected_sum)

    def test_mask_out_pad_token(self):
        # Define the inputs
        input_ids = torch.tensor([[50256]])
        # log_probs = torch.tensor([
        #     [50256, 50256, 50256, 50256, 50256],  # Sequence with EOS token and pad tokens
        #     [50256, 0.7, 0.8, 0.9, 1.0]   # S
        # ])
        sequences = torch.tensor(
            [
                [50256, 50256, 50256, 50256, 50256],
                [50256, 13, 198, 198, 13],
                [50256, 82, 8, 50256, 50256],
                [50256, 12, 49368, 13, 50256],
            ]
        )
        # get logits from model
        logits = get_logits(model, sequences)
        # print("logits: ", logits)
        index = create_index_tensor(sequences, input_ids)

        # get probs from model
        # logprobs = get_logprobs(logits, index, top_k=None, pad_token_id=tokenizer.pad_token_id)
        # print("logprobs: ", logprobs)
        # logprobs = get_logprobs(logits, index, top_k=100, pad_token_id=tokenizer.pad_token_id)
        # print("logprobs with top-k: ", logprobs)
        logprobs = torch.tensor(
            [
                [-6.2530e00, -1.3405e01, -1.3767e01, -1.3907e01],
                [-4.4828e00, -7.3980e-01, -6.6570e-03, -1.9486e00],
                [-6.2220e00, -6.2684e00, -4.5776e00, -1.2446e01],
                [-4.8731e00, -6.6102e00, -6.2977e00, -3.4066e00],
            ]
        )

        inf = float("-inf")
        logprobs_with_top_k = torch.tensor(
            [
                [-5.5133e00, -inf, -inf, -inf],
                [-3.7431e00, -5.0686e-01, -4.7710e-03, -1.3873e00],
                [-5.4823e00, -5.5139e00, -3.6154e00, -inf],
                [-4.1333e00, -5.4711e00, -5.7164e00, -3.0604e00],
            ]
        )

        expected_output = torch.tensor(
            [
                [-5.5133e00, 0, 0, 0],  # Sequence with all but the first EOS masked
                [
                    -3.7431e00,
                    -5.0686e-01,
                    -4.7710e-03,
                    -1.3873e00,
                ],  # Second sequence with no tokens masked
                [
                    -5.4823e00,
                    -5.5139e00,
                    -3.6154e00,
                    0,
                ],  # Third sequence with all but the last PAD masked
                [
                    -4.1333e00,
                    -5.4711e00,
                    -5.7164e00,
                    -3.0604e00,
                ],  # Fourth sequence with no tokens masked
            ]
        )

        # Run the mask_out_pad_token function
        output_logprobs = mask_out_pad_token(logprobs.clone(), index, tokenizer.pad_token_id)
        output_logprobs_with_top_k = mask_out_pad_token(logprobs_with_top_k.clone(), index, tokenizer.pad_token_id)

        # Check if the output matches the expected_output
        self.assertTrue(torch.equal(output_logprobs, expected_output))
        self.assertTrue(torch.equal(output_logprobs_with_top_k, expected_output))

    # TODO: add get_logprobs test


if __name__ == "__main__":
    # Load pre-trained model tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Set the padding side to the left
    tokenizer.padding_side = "left"
    # Load pre-trained model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # Set the model to evaluation mode
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    unittest.main()
