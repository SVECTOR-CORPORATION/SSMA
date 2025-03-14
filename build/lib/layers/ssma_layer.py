import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMALayer(nn.Module):
    """
    Core SSMA Layer that implements:
      - Sparse State Transitions
      - Low-Rank Factorized Interactions
      - Hierarchical Memory via LRU update
    """
    def __init__(self, d_model, r=64, m=256, top_k=32):
        super(SSMALayer, self).__init__()
        self.d_model = d_model
        self.r = r
        self.m = m
        self.top_k = top_k

        # Low-rank projections
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
        self.register_buffer('init_state', torch.zeros(1, m))
        self.gamma = nn.Parameter(torch.tensor(0.9))

    def top_k_gate(self, x):
        # x: (batch, m)
        topk_vals, _ = torch.topk(x, self.top_k, dim=-1)
        threshold = topk_vals[:, -1].unsqueeze(-1)
        mask = (x >= threshold).float()
        return x * mask

    def forward(self, x, state=None):
        """
        x: (batch, seq_len, d_model)
        state: (batch, m); if None, initialized to zeros.
        Returns:
            outputs: (batch, seq_len, d_model)
            memory: (batch, m)
        """
        batch_size, seq_len, _ = x.size()
        if state is None:
            state = self.init_state.expand(batch_size, self.m)
        outputs = []
        memory = state

        for t in range(seq_len):
            xt = x[:, t, :]  # (B, d_model)
            Ux = self.U_proj(xt)
            Vx = self.V_proj(xt)
            ffn_out = self.ffn(xt)
            x_proj = self.W_in(xt)
            state_update = F.relu(torch.matmul(state, self.W_state.weight.T) + x_proj)
            state_update = self.top_k_gate(state_update)
            memory = self.gamma * memory + (1 - self.gamma) * state_update
            state = state_update
            out = xt + ffn_out
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, memory
