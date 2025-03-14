import torch.nn as nn

from .ssma_layer import SSMALayer


class HybridSSMALayer(nn.Module):
    """
    Hybrid SSMA Layer that can toggle between standard self-attention and SSMA.
    Useful for ablation studies and to create hybrid models.
    """
    def __init__(self, d_model, n_heads, use_attention=False):
        super(HybridSSMALayer, self).__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(d_model, n_heads)
        else:
            self.ssma_layer = SSMALayer(d_model, r=64, m=256)
    
    def forward(self, x):
        if self.use_attention:
            # x shape should be (seq_len, batch, d_model) for nn.MultiheadAttention
            return self.attention(x, x, x)[0]
        else:
            return self.ssma_layer(x)
