import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMALayer(nn.Module):
    """
    Single SSMA layer that updates a fixed-size state with sparse gating and integrates it into a feed-forward path.
    """
    def __init__(self, d_model, r=64, m=256, top_k=32):
        super(SSMALayer, self).__init__()
        self.d_model = d_model  # Model hidden dimension
        self.r = r              # Low-rank projection dimension
        self.m = m              # State and memory size (fixed)
        self.top_k = top_k      # Number of top activations to retain

        # Low-rank projection for interaction
        self.U_proj = nn.Linear(d_model, r, bias=False)
        self.V_proj = nn.Linear(d_model, r, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        # Projections for state update
        self.W_in = nn.Linear(d_model, m)
        self.W_state = nn.Linear(m, m, bias=False)

        # Initialize state buffer (will be expanded per batch)
        self.register_buffer('init_state', torch.zeros(1, m))
        # Decay factor for LRU memory update (learnable)
        self.gamma = nn.Parameter(torch.tensor(0.9))

    def top_k_gate(self, x):
        """
        Retains only the top-k values per row.
        x: Tensor of shape (batch, m)
        """
        topk_vals, _ = torch.topk(x, self.top_k, dim=-1)
        threshold = topk_vals[:, -1].unsqueeze(-1)
        mask = (x >= threshold).float()
        return x * mask

    def forward(self, x, state=None):
        """
        x: Tensor of shape (batch, seq_len, d_model)
        state: Tensor of shape (batch, m). If None, initializes to zeros.
        Returns:
            outputs: Tensor of shape (batch, seq_len, d_model)
            memory: Tensor of shape (batch, m)
        """
        batch_size, seq_len, _ = x.size()
        if state is None:
            state = self.init_state.expand(batch_size, self.m)

        outputs = []
        memory = state  # Initial memory is the starting state

        for t in range(seq_len):
            xt = x[:, t, :]  # (B, d_model)
            # Compute low-rank interaction
            Ux = self.U_proj(xt)  # (B, r)
            Vx = self.V_proj(xt)  # (B, r)
            # Optionally, you could combine Ux and Vx in various ways;
            # here we simply compute an outer product and flatten (for demonstration)
            interaction = torch.bmm(Ux.unsqueeze(2), Vx.unsqueeze(1))
            interaction_flat = interaction.view(batch_size, -1)
            ffn_out = self.ffn(xt)

            # Project input into state space
            x_proj = self.W_in(xt)  # (B, m)
            # Update state: combine previous state and new projected input
            state_update = F.relu(torch.matmul(state, self.W_state.weight.T) + x_proj)
            state_update = self.top_k_gate(state_update)  # Enforce sparsity via Top-k gating

            # Update hierarchical memory (LRU update)
            memory = self.gamma * memory + (1 - self.gamma) * state_update
            state = state_update

            # Residual connection with FFN output
            output = xt + ffn_out
            outputs.append(output.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # (B, seq_len, d_model)
        return outputs, memory

class SSMA_Model(nn.Module):
    """
    SSMA-based model with multiple layers.
    """
    def __init__(self, d_model=512, num_layers=6, r=64, m=256, top_k=32, vocab_size=10000, max_seq_len=512):
        super(SSMA_Model, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([SSMALayer(d_model, r, m, top_k) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len) of token indices.
        Returns:
            logits: Tensor of shape (batch, seq_len, vocab_size)
            memories: List of memory states from each layer.
        """
        batch_size, seq_len = x.size()
        emb = self.embedding(x)  # (B, seq_len, d_model)
        emb = emb + self.pos_embedding[:, :seq_len, :]
        out = emb
        memories = []
        for layer in self.layers:
            out, memory = layer(out)
            memories.append(memory)
        out = self.layer_norm(out)
        logits = self.fc_out(out)
        return logits, memories

# For testing the model directly
if __name__ == '__main__':
    model = SSMA_Model()
    dummy_input = torch.randint(0, 10000, (2, 50))  # Batch size 2, sequence length 50
    logits, memories = model(dummy_input)
    print("Logits shape:", logits.shape)       # Expected: (2, 50, vocab_size)
    print("Memory from last layer shape:", memories[-1].shape)  # Expected: (2, m)
