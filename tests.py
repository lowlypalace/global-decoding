import unittest
import torch

from sequence_probability import (
    top_k_filtering,
    create_index_tensor,
    sum_logprobs,
    mask_out_pad_token,
)

# TODO: Run tests on commits


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
    def test_index_with_pad_tokens(self):
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
        pad_token_id = 50256
        # First sequence with all but the first EOS masked
        # Second sequence with no tokens masked
        # Third sequence with all but the last PAD masked
        # Fourth sequence with no tokens masked
        sequences = torch.tensor(
            [
                [50256, 50256, 50256, 50256, 50256],
                [50256, 13, 198, 198, 13],
                [50256, 82, 8, 50256, 50256],
                [50256, 12, 49368, 13, 50256],
            ]
        )
        index = create_index_tensor(sequences, input_ids)
        logprobs = torch.tensor(
            [
                [-6.2530, -13.405, -13.767, -13.907],
                [-4.4828, -0.7398, -0.0067, -1.9486],
                [-6.2220, -6.2684, -4.5776, -12.446],
                [-4.8731, -6.6102, -6.2977, -3.4066],
            ]
        )
        inf = float("-inf")
        logprobs_with_top_k = torch.tensor(
            [
                [-5.5133, -inf, -inf, -inf],
                [-3.7431, -0.5069, -0.0048, -1.3873],
                [-5.4823, -5.5139, -3.6154, -inf],
                [-4.1333, -5.4711, -5.7164, -3.0604],
            ]
        )
        expected_output_logprobs = torch.tensor(
            [
                [-6.2530, 0, 0, 0],
                [-4.4828, -0.7398, -0.0067, -1.9486],
                [-6.2220, -6.2684, -4.5776, 0],
                [-4.8731, -6.6102, -6.2977, -3.4066],
            ]
        )
        expected_output_logprobs_with_top_k = torch.tensor(
            [
                [-5.5133, 0, 0, 0],
                [-3.7431, -0.5069, -0.0048, -1.3873],
                [-5.4823, -5.5139, -3.6154, 0],
                [-4.1333, -5.4711, -5.7164, -3.0604],
            ]
        )
        # Run the mask_out_pad_token function
        output_logprobs = mask_out_pad_token(logprobs.clone(), index, pad_token_id)
        output_logprobs_with_top_k = mask_out_pad_token(
            logprobs_with_top_k.clone(), index, pad_token_id
        )
        # Check if the output matches the expected_output
        self.assertTrue(torch.equal(output_logprobs, expected_output_logprobs))
        self.assertTrue(
            torch.equal(output_logprobs_with_top_k, expected_output_logprobs_with_top_k)
        )

    # TODO: add get_logprobs test


if __name__ == "__main__":
    unittest.main()
