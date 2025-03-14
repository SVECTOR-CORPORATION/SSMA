import torch
import torch.nn.functional as F


def ternary_quantize(weight):
    """
    Quantize weights to ternary values {-1, 0, +1} using the sign function.
    """
    return weight.sign()

def quantization_loss(weight):
    """
    Compute the quantization loss to encourage weights to be close to {-1, 0, +1}.
    """
    return F.mse_loss(weight, weight.sign())
