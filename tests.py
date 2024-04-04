import unittest
import torch

from utils import top_k_batch_filtering


class TestTopKFiltering(unittest.TestCase):

    def test_single_value(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        top_k = 1
        filtered_logits = top_k_batch_filtering(logits, top_k)
        self.assertTrue(torch.allclose(filtered_logits, torch.tensor([[-float('inf'), -float('inf'), 3.0]])))

    def test_all_values(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        top_k = 3
        filtered_logits = top_k_batch_filtering(logits, top_k)
        self.assertTrue(torch.allclose(filtered_logits, logits))

    def test_batched_logits(self):
        logits = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        top_k = 2
        filtered_logits = top_k_batch_filtering(logits, top_k)
        expected_output = torch.tensor([
            [-float('inf'), 2.0, 3.0],
            [-float('inf'), 5.0, 6.0]
        ])
        self.assertTrue(torch.allclose(filtered_logits, expected_output))

    def test_top_k_filtering_on_large_batch(self):
        # Testing on a larger batch to ensure the function scales
        batch_size = 64
        sequence_length = 10
        vocab_size = 100
        top_k = 5

        logits = torch.randn(batch_size, sequence_length, vocab_size)
        filtered_logits = top_k_batch_filtering(logits, top_k)

        # For each item in the batch, check that there are exactly top_k values not equal to -inf
        for i in range(batch_size):
            for j in range(sequence_length):
                self.assertEqual(torch.isfinite(filtered_logits[i, j]).sum().item(), top_k)

if __name__ == '__main__':
    unittest.main()
