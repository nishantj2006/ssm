import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

MODEL_DIR = str(Path(__file__).resolve().parents[1])
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)
from ssm_scan import selective_scan

# 1. THE NEW SELECTIVE SSM LAYER (Mamba-Style)
class SingleHeadSSMLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.log_A = nn.Parameter(torch.randn(dim) * 0.02)
        self.dt_proj = nn.Linear(dim, 1) 
        
        # The Selective Gates
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
        
        gated_x = B * x  # Input Gate
        return selective_scan(gated_x, C, dt, self.log_A, x_in, self.scan_backend)

# 2. ROTARY POSITIONAL EMBEDDINGS (RoPE)
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

def rotate_half(x):
    d = x.shape[-1]
    return torch.cat((-x[..., d // 2:], x[..., :d // 2]), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # Expand cos/sin to (1, 1, T, D) matching (B, num_heads, T, head_dim)
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out

# 3. CAUSAL ATTENTION WITH RoPE
class CausalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_attn = nn.Linear(dim, dim * 3)
        self.c_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        x_in = x
        x = self.norm(x)
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE relative positional embeddings
        cos, sin = self.rotary(T, device=x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # PyTorch's ultra-fast Flash Attention built-in
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y) + x_in

# 4. FEED FORWARD NETWORK
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

# 5. THE HYBRID BLOCK MANAGER
class HybridBlock(nn.Module):
    def __init__(self, dim, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        if self.use_attention:
            self.attn = CausalAttention(dim)
        else:
            self.ssm = SingleHeadSSMLayer(dim)
            
        self.ffn = FeedForward(dim)

    def forward(self, x):
        if self.use_attention:
            x = self.attn(x)
        else:
            x = self.ssm(x)
            
        x = self.ffn(x)
        return x

# 6. THE MASTER HYBRID LANGUAGE MODEL
class HybridLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers=8, attn_layers=None, attn_every=None):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Inward anchors: By default, place attention at Layers 3 and 6 (indices 2 and 5)
        # keeping Layers 7 and 8 as SSM to preserve temporal resolution before the classifier head.
        if attn_layers is not None:
            use_attn_set = set(attn_layers)
        elif attn_every is not None:
            use_attn_set = {i for i in range(num_layers) if (i + 1) % attn_every == 0}
        else:
            use_attn_set = {2, 5}
            
        self.layers = nn.ModuleList([
            HybridBlock(dim, use_attention=(i in use_attn_set)) 
            for i in range(num_layers)
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
