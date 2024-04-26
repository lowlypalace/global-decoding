import unittest
import torch

from generate_sequences import (
    pad_sequences,
)


class TestGenerateSequences(unittest.TestCase):
    def test_pad_sequences(self):
        # Create a tensor of sequences with varying lengths
        sequences = torch.tensor([[1, 2, 3], [1, 2], [1]])
        pad_token_id = 50256
        max_length = 5

        # Expected output: all sequences should be padded to length 5
        expected_output = torch.tensor([
            [1, 2, 3, 0, 0],
            [1, 2, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ])

        # Pad the sequences
        padded_sequences = pad_sequences(sequences, pad_token_id, max_length)

        # Assert that the output is as expected
        self.assertTrue(torch.equal(padded_sequences, expected_output),
                        "The padded sequences do not match the expected output.")


if __name__ == "__main__":
    unittest.main()
