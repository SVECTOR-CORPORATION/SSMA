import unittest

import torch
from ssma.model import SSMA_Model


class TestSSMAModel(unittest.TestCase):
    def test_forward_pass(self):
        model = SSMA_Model(d_model=512, num_layers=2, r=64, m=256, top_k=32, vocab_size=10000, max_seq_len=50)
        x = torch.randint(0, 10000, (2, 50))
        logits, memories = model(x)
        self.assertEqual(logits.shape, (2, 50, 10000))
        self.assertEqual(len(memories), 2)

if __name__ == "__main__":
    unittest.main()
