import unittest
import torch

from utils import top_k_batch_filtering
from global_decoding_hf import (
    create_index_tensor,
    get_logits,
    get_original_logprobs,
    get_proposal_logprobs,
    get_sequence_probs,
)

from torch.nn.functional import log_softmax


class TestTopKFiltering(unittest.TestCase):

    def test_single_value(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        top_k = 1
        filtered_logits = top_k_batch_filtering(logits, top_k)
        self.assertTrue(
            torch.allclose(
                filtered_logits, torch.tensor([[-float("inf"), -float("inf"), 3.0]])
            )
        )

    def test_all_values(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        top_k = 3
        filtered_logits = top_k_batch_filtering(logits, top_k)
        self.assertTrue(torch.allclose(filtered_logits, logits))

    def test_batched_logits(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        top_k = 2
        filtered_logits = top_k_batch_filtering(logits, top_k)
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
        filtered_logits = top_k_batch_filtering(logits, top_k)

        # For each item in the batch, check that there are exactly top_k values not equal to -inf
        for i in range(batch_size):
            for j in range(sequence_length):
                self.assertEqual(
                    torch.isfinite(filtered_logits[i, j]).sum().item(), top_k
                )


class MockModel(torch.nn.Module):
    def __init__(self, output_logits):
        super().__init__()
        self.output_logits = output_logits

    def forward(self, input_ids, return_dict=True):
        # Mock output to resemble a model's output
        return {"logits": self.output_logits}


class TestSequenceProbability(unittest.TestCase):

    def test_create_index_tensor(self):
        generated_ids = torch.tensor(
            [
                [101, 102, 103, 0],
                [101, 104, 105, 106],
            ]
        )
        expected_index = torch.tensor([[[1], [2], [3]], [[1], [2], [3]]])
        index = create_index_tensor(generated_ids)
        self.assertTrue(torch.equal(index, expected_index))

    def test_get_logits(self):
        mock_output_logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        model = MockModel(output_logits=mock_output_logits)
        generated_ids = torch.tensor([[101, 102], [101, 103]])
        logits = get_logits(model, generated_ids)
        self.assertTrue(torch.equal(logits, mock_output_logits[:, :-1]))

    # def test_get_original_logprobs(self):
    #     logits = torch.tensor([[[-1.0, 0.0, 1.0]]])
    #     index = torch.tensor([[[2]]])
    #     original_logprobs = get_original_logprobs(logits, index)
    #     expected_logprobs = log_softmax(logits, dim=-1).squeeze(-2)
    #     self.assertTrue(torch.allclose(original_logprobs, expected_logprobs))

    # def test_get_proposal_logprobs(self):
    #     logits = torch.tensor([[[-1.0, 0.0, 1.0]]])
    #     top_k = 2
    #     index = torch.tensor([[[2]]])
    #     proposal_logprobs = get_proposal_logprobs(logits, top_k, index)
    #     filtered_logits = top_k_batch_filtering(logits, top_k)
    #     expected_logprobs = log_softmax(filtered_logits, dim=-1).squeeze(-2)
    #     self.assertTrue(torch.allclose(proposal_logprobs, expected_logprobs))

    # def test_get_sequence_probs(self):
    #     mock_output_logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    #     model = MockModel(output_logits=mock_output_logits)
    #     generated_ids = torch.tensor([[101, 102], [101, 103]])
    #     top_k = 2
    #     original_logprob_sum, proposal_logprob_sum = get_sequence_probs(
    #         model, generated_ids, top_k
    #     )

    #     self.assertIsInstance(original_logprob_sum, float)
    #     self.assertIsInstance(proposal_logprob_sum, float)


if __name__ == "__main__":
    unittest.main()
