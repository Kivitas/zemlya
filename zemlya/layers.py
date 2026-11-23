"""Building blocks: RMSNorm, RoPE, grouped-query attention, SwiGLU, transformer stack."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device, dtype):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Multi-head attention with GQA + RoPE. `causal=False` for the predictor."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, causal: bool, rope_base: float,
                 use_rope: bool = True):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.causal, self.rope_base = n_heads, n_kv_heads, causal, rope_base
        self.use_rope = use_rope
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            cos, sin = rope_cos_sin(T, self.head_dim, self.rope_base, x.device, q.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.o(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float):
        super().__init__()
        hidden = int(2 * mult * dim / 3)
        hidden = (hidden + 63) // 64 * 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, ffn_mult, causal, rope_base):
        super().__init__()
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads, causal, rope_base)
        self.ffn = SwiGLU(dim, ffn_mult)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, dim, n_layers, n_heads, n_kv_heads, ffn_mult, causal, rope_base):
        super().__init__()
        self.blocks = nn.ModuleList(
            Block(dim, n_heads, n_kv_heads, ffn_mult, causal, rope_base) for _ in range(n_layers)
        )
        self.norm = RMSNorm(dim)
        self.apply(self._init)
        # depth-scaled init on residual output projections
        for b in self.blocks:
            nn.init.normal_(b.attn.o.weight, std=0.02 / (2 * n_layers) ** 0.5)
            nn.init.normal_(b.ffn.w2.weight, std=0.02 / (2 * n_layers) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


class MLP(nn.Module):
    def __init__(self, i: int, h: int, o: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


class VariateBlock(nn.Module):
    """Cross-series attention at a fixed time step (TimesFM-3 style). Channels are an unordered
    set, so no positional encoding; attention-only (no FFN) to stay cheap."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads, causal=False, rope_base=1.0, use_rope=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, N, D)
        B, C, N, D = x.shape
        y = x.permute(0, 2, 1, 3).reshape(B * N, C, D)
        y = y + self.attn(self.norm(y))
        return y.reshape(B, N, C, D).permute(0, 2, 1, 3)
