import unittest
import torch

from src.sequences.sequences_probs import (
    top_k_filtering,
    top_p_filtering,
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


class TestTopPFiltering(unittest.TestCase):
    def test_single_value(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        top_p = 0.9  # Choose a top_p that would include the last logit when softmax is applied
        filtered_logits = top_p_filtering(logits, top_p)
        # Only the last logit should be kept, others should be set to -inf
        self.assertTrue(
            torch.allclose(
                filtered_logits, torch.tensor([[-float("inf"), -float("inf"), 3.0]])
            )
        )

    def test_all_values(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        top_p = 1.0  # Setting top_p to 1 should keep all values
        filtered_logits = top_p_filtering(logits, top_p)
        self.assertTrue(torch.allclose(filtered_logits, logits))

    def test_batched_logits(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        top_p = 0.8  # Adjust top_p to filter more strictly
        filtered_logits = top_p_filtering(logits, top_p)
        expected_output = torch.tensor(
            [[-float("inf"), -float("inf"), 3.0], [-float("inf"), -float("inf"), 6.0]]
        )
        self.assertTrue(torch.allclose(filtered_logits, expected_output))

    def test_top_p_filtering_on_large_batch(self):
        # Testing on a larger batch to ensure the function scales
        batch_size = 64
        sequence_length = 10
        vocab_size = 100
        top_p = 0.9

        logits = torch.randn(batch_size, sequence_length, vocab_size)
        filtered_logits = top_p_filtering(logits, top_p)

        # Check that the cumulative probability of non--inf values does not exceed top_p
        for i in range(batch_size):
            for j in range(sequence_length):
                finite_logits = filtered_logits[i, j][torch.isfinite(filtered_logits[i, j])]
                probs = torch.softmax(finite_logits, dim=0)
                self.assertTrue(torch.cumsum(probs, dim=0)[-1] <= top_p)

if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
