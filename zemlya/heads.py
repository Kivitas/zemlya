"""Decision/prediction heads that read latents (never raw inputs)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ZemlyaConfig
from .layers import MLP
from .losses import symlog


class ForecastHead(nn.Module):
    """Latent -> quantile forecast of one patch (TimesFM-style probabilistic output)."""

    def __init__(self, cfg: ZemlyaConfig):
        super().__init__()
        self.cfg = cfg
        self.net = MLP(cfg.d_model, cfg.head_hidden, cfg.patch_len * cfg.channels * len(cfg.quantiles))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, H, _ = z.shape
        c = self.cfg
        out = self.net(z).view(B, H, c.patch_len, c.channels, len(c.quantiles))
        # monotone quantiles by construction (no quantile crossing)
        return torch.cat([out[..., :1], out[..., :1] + torch.cumsum(F.softplus(out[..., 1:]), -1)], -1)


class CriticHead(nn.Module):
    """Ensemble of reward models r(z,a) and value models V(z), both in symlog space (DreamerV3).
    Ensemble disagreement is the epistemic-uncertainty signal used for pessimistic planning."""

    def __init__(self, cfg: ZemlyaConfig):
        super().__init__()
        K, d, h, A = cfg.critic_ensemble, cfg.d_model, cfg.head_hidden, cfg.action_dim
        self.rew = nn.ModuleList(MLP(d + A, h, 1) for _ in range(K))
        self.val = nn.ModuleList(MLP(d, h, 1) for _ in range(K))

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:   # -> (K,B,H) symlog
        x = torch.cat([z, a], -1)
        return torch.stack([n(x).squeeze(-1) for n in self.rew])

    def value(self, z: torch.Tensor) -> torch.Tensor:                      # -> (K,B,H) symlog
        return torch.stack([n(z).squeeze(-1) for n in self.val])


class PolicyHead(nn.Module):
    """Amortized, return-conditioned behavior policy (WHAM's 'policy mode' + Decision-Transformer
    style hindsight conditioning). last context latent (+ desired return) -> Gaussian per patch."""

    def __init__(self, cfg: ZemlyaConfig):
        super().__init__()
        self.cfg = cfg
        self.ret_embed = nn.Linear(1, cfg.d_model)
        self.null = nn.Parameter(torch.zeros(cfg.d_model))
        self.net = MLP(cfg.d_model, cfg.head_hidden, 2 * cfg.max_horizon_patches * cfg.action_dim)

    def forward(self, z_last: torch.Tensor, horizon: int, ret=None, keep=None):
        """ret: None | float | (B,) desired discounted return (real units); keep:(B,) bool mask of
        which rows use the return (others use the null token) -- used for conditioning dropout."""
        c, B = self.cfg, z_last.shape[0]
        null = self.null.expand(B, -1)
        if ret is None:
            e = null
        else:
            r = torch.as_tensor(ret, device=z_last.device, dtype=z_last.dtype).expand(B)
            e = self.ret_embed(symlog(r)[:, None])
            if keep is not None:
                e = torch.where(keep[:, None], e, null)
        out = self.net(z_last + e).view(-1, 2, c.max_horizon_patches, c.action_dim)[:, :, :horizon]
        return out[:, 0], out[:, 1].clamp(-5.0, 1.0)
