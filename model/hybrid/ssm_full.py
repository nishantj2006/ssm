import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
    def forward(self, x):
        x_in = x
        x = self.norm(x)
        b, seq, d = x.shape
        
        A_continuous = -torch.exp(self.log_A) 
        dt = F.softplus(self.dt_proj(x))      
        
        B = self.B_proj(x) 
        C = self.C_proj(x) 
        
        log_A = dt * A_continuous             
        gated_x = B * x  # Input Gate
        
        cumsum_A = torch.cumsum(log_A, dim=1)
        log_M = cumsum_A.unsqueeze(2) - cumsum_A.unsqueeze(1) 
        
        indices = torch.arange(seq, device=x.device)
        mask = indices[:, None] >= indices[None, :]
        log_M = log_M.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
        
        log_M = torch.clamp(log_M, max=20.0)

        M = torch.exp(log_M)
        state = torch.einsum('b t j d, b j d -> b t d', M, gated_x)
        
        out = state * C  # Output Gate
        return out + x_in

# 2. STANDARD CAUSAL ATTENTION (The logic engine)
class CausalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_attn = nn.Linear(dim, dim * 3)
        self.c_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_in = x
        x = self.norm(x)
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # PyTorch's ultra-fast Flash Attention built-in
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y) + x_in

# 3. FEED FORWARD NETWORK
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

# 4. THE HYBRID BLOCK MANAGER
class HybridBlock(nn.Module):
    def __init__(self, dim, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        # This block will either be an Attention layer or an SSM layer
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

# 5. THE MASTER HYBRID LANGUAGE MODEL
class HybridLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers=8, attn_every=2):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # This loop interlacing the layers! e.g., SSM -> Attn -> SSM -> Attn
        self.layers = nn.ModuleList([
            HybridBlock(dim, use_attention=((i + 1) % attn_every == 0)) 
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