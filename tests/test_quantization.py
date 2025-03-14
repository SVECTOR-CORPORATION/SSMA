import unittest

import torch
from ssma.quantization.quantize import quantization_loss, ternary_quantize


class TestQuantization(unittest.TestCase):
    def test_ternary_quantize(self):
        weight = torch.tensor([[0.5, -0.7, 0.0], [1.2, -0.3, 0.0]])
        quantized = ternary_quantize(weight)
        self.assertTrue(torch.all(quantized.abs() <= 1))
    
    def test_quantization_loss(self):
        weight = torch.tensor([[0.5, -0.7, 0.0], [1.2, -0.3, 0.0]])
        loss = quantization_loss(weight)
        self.assertGreaterEqual(loss.item(), 0)

if __name__ == "__main__":
    unittest.main()
