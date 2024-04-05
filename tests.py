import unittest
import torch

from sequence_probability import top_k_filtering, create_index_tensor, sum_logprobs


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

    # TODO: add get_logprobs test


if __name__ == "__main__":
    unittest.main()
