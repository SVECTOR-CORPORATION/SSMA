import unittest

import torch
from ssma.layers.ssma_layer import SSMALayer


class TestSSMALayer(unittest.TestCase):
    def test_top_k_gate(self):
        layer = SSMALayer(d_model=512, r=64, m=256, top_k=10)
        x = torch.rand(2, 256)
        gated = layer.top_k_gate(x)
        # Ensure that at most 10 values per row are non-zero
        nonzero_counts = (gated != 0).sum(dim=1)
        for count in nonzero_counts:
            self.assertLessEqual(count.item(), 10)

if __name__ == "__main__":
    unittest.main()
