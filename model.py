import torch.nn as nn

from .layers.ssma_layer import SSMALayer


class SSMA_Model(nn.Module):
    """
    Base SSMA Model.
    
    Users can build custom transformer-like models by instantiating this class
    and modifying or extending it.
    """
    def __init__(self, d_model=512, num_layers=6, r=64, m=256, top_k=32, vocab_size=10000, max_seq_len=512):
        super(SSMA_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([SSMALayer(d_model, r, m, top_k) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len) containing token indices.
        Returns:
            logits: Tensor of shape (batch, seq_len, vocab_size)
            memories: List of memory states from each layer.
        """
        batch_size, seq_len = x.size()
        emb = self.embedding(x)
        emb = emb + self.pos_embedding[:, :seq_len, :]
        out = emb
        memories = []
        for layer in self.layers:
            out, memory = layer(out)
            memories.append(memory)
        out = self.layer_norm(out)
        logits = self.fc_out(out)
        return logits, memories
