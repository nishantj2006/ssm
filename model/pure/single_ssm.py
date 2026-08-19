import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. THE SELECTIVE SSM LAYER
class SingleHeadSSMLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.log_A = nn.Parameter(torch.randn(dim) * 0.02)
        self.dt_proj = nn.Linear(dim, 1) 
        
        self.B_proj = nn.Linear(dim, dim)
        self.C_proj = nn.Linear(dim, dim)
        
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        
    def forward(self, x):
        x_in = x
        x = self.norm(x)
        b, seq, d = x.shape
        
        A_continuous = -torch.exp(self.log_A) 
        dt = F.softplus(self.dt_proj(x))      
        
        B = self.B_proj(x) 
        C = self.C_proj(x) 
        
        log_A = dt * A_continuous             
        gated_x = B * x  
        
        cumsum_A = torch.cumsum(log_A, dim=1)
        log_M = cumsum_A.unsqueeze(2) - cumsum_A.unsqueeze(1) 
        
        indices = torch.arange(seq, device=x.device)
        mask = indices[:, None] >= indices[None, :]
        log_M = log_M.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
        
        log_M = torch.clamp(log_M, max=20.0)

        M = torch.exp(log_M)
        state = torch.einsum('b t j d, b j d -> b t d', M, gated_x)
        
        out = state * C  
        return out + x_in

# 2. FEED FORWARD NETWORK
class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.net(self.norm(x))

# 3. PURE SSM BLOCK
class PureSSMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ssm = SingleHeadSSMLayer(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        x = self.ssm(x)
        x = self.ffn(x)
        return x

# 4. MASTER PURE SSM MODEL
class PureSSMLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers=6):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Every single layer is now an SSM. Zero Attention.
        self.layers = nn.ModuleList([
            PureSSMBlock(dim) for _ in range(num_layers)
        ])
            
        self.final_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.classifier(x)
        return logits