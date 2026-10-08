import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import sys
from pathlib import Path

MODEL_DIR = str(Path(__file__).resolve().parents[1])
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)
from ssm_scan import selective_scan

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
        self.scan_backend = "auto"
        
    def forward(self, x):
        x_in = x
        x = self.norm(x)
        b, seq, d = x.shape
        
        dt = F.softplus(self.dt_proj(x))      
        
        B = self.B_proj(x) 
        C = self.C_proj(x) 
        
        gated_x = B * x  
        return selective_scan(gated_x, C, dt, self.log_A, x_in, self.scan_backend)

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
        self.gradient_checkpointing = False
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Every single layer is now an SSM. Zero Attention.
        self.layers = nn.ModuleList([
            PureSSMBlock(dim) for _ in range(num_layers)
        ])
            
        self.final_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, vocab_size, bias=False)

    def forward_hidden(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.final_norm(x)

    def forward(self, input_ids):
        return self.classifier(self.forward_hidden(input_ids))
