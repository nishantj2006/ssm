import torch
import torch.nn as nn
import torch.nn.functional as F

class PureInferenceCache:
    def __init__(self, batch_size, dim, device):
        self.ssm_states = {}  
        self.batch_size = batch_size
        self.dim = dim
        self.device = device

    def get_ssm_state(self, layer_idx):
        if layer_idx not in self.ssm_states:
            self.ssm_states[layer_idx] = torch.zeros(
                self.batch_size, self.dim, self.dim, device=self.device
            )
        return self.ssm_states[layer_idx]

    def update_ssm_state(self, layer_idx, new_state):
        self.ssm_states[layer_idx] = new_state

class MultiHeadSSMLayer(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # Ensure dimensions can be split cleanly across heads
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads"
        self.head_dim = dim // num_heads
        
        # We now track separate A-matrices for each head
        self.log_A = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)
        self.dt_proj = nn.Linear(dim, num_heads) 
        self.norm = nn.LayerNorm((dim, dim), eps=1e-5)
        
    def forward(self, x):
        x_in = x
        x = self.norm(x)
        b, seq, d, _ = x.shape
        
        A_continuous = -torch.exp(self.log_A) # Shape: [Heads, Head_Dim]
        dt = F.softplus(self.dt_proj(x)).mean(dim=-1) # Shape: [B, Seq, Heads]
        
        # Reshape to apply multi-head math
        A_expanded = A_continuous.view(1, 1, self.num_heads, self.head_dim)
        dt_expanded = dt.unsqueeze(-1)
        
        log_A = dt_expanded * A_expanded 
        log_A = log_A.view(b, seq, d) # Flatten back to match original dim
        
        cumsum_A = torch.cumsum(log_A, dim=1)
        log_M = cumsum_A.unsqueeze(2) - cumsum_A.unsqueeze(1) 
        
        indices = torch.arange(seq, device=x.device)
        mask = indices[:, None] >= indices[None, :]
        log_M = log_M.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
        
        M = torch.exp(log_M)
        out = torch.einsum('b t j r, b j r c -> b t r c', M, x)
        return out + x_in

class MatrixFeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        input_dim = dim * dim
        hidden_dim = input_dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.norm = nn.LayerNorm((dim, dim))

    def forward(self, x):
        residual = x
        x = self.norm(x)
        b, s, d, _ = x.shape
        x = x.view(b, s, -1)
        x = self.net(x)
        x = x.view(b, s, d, d)
        return residual + x

class PureSSMBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.ssm = MultiHeadSSMLayer(dim, num_heads=num_heads)
        self.ffn = MatrixFeedForward(dim)

    def forward(self, x, cache=None, layer_idx=None):
        # We only need the parallel forward pass for training right now
        x = self.ssm(x)
        x = self.ffn(x)
        return x

class PureSSMLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers=4, num_heads=4):
        super().__init__()
        self.dim = dim
        self.embed_dim = dim * dim
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        
        self.layers = nn.ModuleList([
            PureSSMBlock(dim, num_heads=num_heads) for _ in range(num_layers)
        ])
            
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, vocab_size, bias=False)

    def forward(self, input_ids, cache=None):
        x = self.embedding(input_ids)
        b, s, _ = x.shape
        x = x.view(b, s, self.dim, self.dim)
        
        for i, layer in enumerate(self.layers):
            x = layer(x, cache=cache, layer_idx=i)
            
        x = x.view(b, s, -1)
        x = self.final_norm(x)
        logits = self.classifier(x)
        return logits