"""Patch tokenizer, instance normalization, and the causal context/target encoder.

Tokens live on a (channel x patch) grid. Blocks alternate TimesFM-3-style:
  * causal temporal attention along patches (per channel), and
  * cross-series (variate) attention across channels at the same patch.
Channel tokens are pooled into one joint latent per patch for the predictor/heads.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from .config import ZemlyaConfig
from .layers import Block, RMSNorm, Transformer, VariateBlock


class Stats(NamedTuple):
    mean: torch.Tensor  # (B,1,C)
    std: torch.Tensor   # (B,1,C)


def normalize(x: torch.Tensor, mask: Optional[torch.Tensor]):
    """Masked per-sample, per-channel instance norm (RevIN style). x:(B,T,C) mask:(B,T) 1=valid."""
    if mask is None:
        mask = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
    m = mask.unsqueeze(-1).to(x.dtype)
    n = m.sum(1, keepdim=True).clamp(min=1.0)
    mean = (x * m).sum(1, keepdim=True) / n
    var = (((x - mean) * m) ** 2).sum(1, keepdim=True) / n
    std = var.sqrt().clamp(min=1e-3)
    return ((x - mean) / std) * m, Stats(mean, std), mask


def patchify(x: torch.Tensor, mask: torch.Tensor, patch_len: int) -> torch.Tensor:
    """(B,T,C),(B,T) -> (B,C,N,2P): per-channel patch values and validity flags."""
    B, T, C = x.shape
    N = T // patch_len
    v = x.reshape(B, N, patch_len, C).permute(0, 3, 1, 2)
    m = mask.reshape(B, N, patch_len)[:, None].expand(B, C, N, patch_len).to(x.dtype)
    return torch.cat([v, m], dim=-1)


class ContextEncoder(nn.Module):
    """Used online and (as an EMA copy) as the target encoder."""

    def __init__(self, cfg: ZemlyaConfig):
        super().__init__()
        d, P, C = cfg.d_model, cfg.patch_len, cfg.channels
        self.cfg = cfg
        self.embed = nn.Sequential(nn.Linear(2 * P, d), nn.SiLU(), nn.Linear(d, d))   # shared over channels
        self.chan_embed = nn.Parameter(torch.randn(C, d) * 0.02)
        # level/scale removed by instance-norm are fed back so absolute-level rewards stay predictable
        self.stat_embed = nn.Linear(2, d)
        self.blocks = nn.ModuleList(
            Block(d, cfg.n_heads, cfg.n_kv_heads, cfg.ffn_mult, True, cfg.rope_base) for _ in range(cfg.n_layers))
        self.variate = nn.ModuleDict({
            str(i): VariateBlock(d, cfg.n_heads, cfg.n_kv_heads)
            for i in range(cfg.n_layers) if C > 1 and (i + 1) % cfg.variate_every == 0})
        self.norm = RMSNorm(d)
        self.apply(Transformer._init)
        for b in self.blocks:
            nn.init.normal_(b.attn.o.weight, std=0.02 / (2 * cfg.n_layers) ** 0.5)
            nn.init.normal_(b.ffn.w2.weight, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    def forward(self, patches: torch.Tensor, stats: Stats) -> torch.Tensor:
        """patches:(B,C,N,2P) -> joint latent per patch (B,N,D)."""
        B, C, N, _ = patches.shape
        m, s = stats.mean.squeeze(1), stats.std.squeeze(1)                           # (B,C)
        feat = torch.stack([torch.sign(m) * torch.log1p(m.abs()), torch.log(s)], -1)  # (B,C,2)
        x = self.embed(patches) + self.stat_embed(feat).unsqueeze(2) + self.chan_embed[None, :, None]
        d = x.shape[-1]
        for i, blk in enumerate(self.blocks):
            x = blk(x.reshape(B * C, N, d)).reshape(B, C, N, d)
            if str(i) in self.variate:
                x = self.variate[str(i)](x)
        return self.norm(x).mean(dim=1)
