"""Action-conditioned latent predictor (the 'P' in JEPA).

Takes context latents + a candidate action per future patch and predicts the *latent* of every
future patch in one parallel pass -- no autoregressive pixel/value decoding, which is what keeps
planning cheap enough for small models.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ZemlyaConfig
from .layers import Transformer


class LatentPredictor(nn.Module):
    def __init__(self, cfg: ZemlyaConfig):
        super().__init__()
        self.cfg = cfg
        d, dp, A = cfg.d_model, cfg.pred_d_model, cfg.action_dim
        self.in_proj = nn.Linear(d, dp)
        self.query = nn.Parameter(torch.randn(cfg.max_horizon_patches, dp) * 0.02)
        self.action_mlp = nn.Sequential(nn.Linear(A, dp), nn.SiLU(), nn.Linear(dp, dp))
        self.backbone = Transformer(dp, cfg.pred_layers, cfg.pred_heads, cfg.pred_kv_heads,
                                    cfg.ffn_mult, causal=False, rope_base=cfg.rope_base)
        self.out_proj = nn.Linear(dp, d)

    def forward(self, z_ctx: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z_ctx:(B,N,D)  actions:(B,H,A) -> predicted future latents (B,H,D)."""
        B, H, _ = actions.shape
        assert H <= self.cfg.max_horizon_patches
        ctx = self.in_proj(z_ctx)
        q = self.query[:H].unsqueeze(0) + self.action_mlp(actions)
        h = self.backbone(torch.cat([ctx, q], dim=1))
        return self.out_proj(h[:, -H:])
