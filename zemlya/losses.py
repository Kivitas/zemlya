"""Losses: latent prediction + anti-collapse regularizers + decision/forecast heads."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 between layer-normalized latents (target is stop-grad from the EMA encoder)."""
    p = F.layer_norm(pred.float(), pred.shape[-1:])
    t = F.layer_norm(target.float(), target.shape[-1:])
    return F.smooth_l1_loss(p, t)


def _flat(z: torch.Tensor, max_rows: int = 4096) -> torch.Tensor:
    z = z.reshape(-1, z.shape[-1]).float()
    if z.shape[0] > max_rows:
        z = z[torch.randperm(z.shape[0], device=z.device)[:max_rows]]
    return z


def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """VICReg variance term: keep each latent dim's std above gamma (prevents collapse)."""
    z = _flat(z)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(gamma - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance term: decorrelate latent dims (prevents dimensional collapse)."""
    z = _flat(z)
    z = z - z.mean(0, keepdim=True)
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return off.pow(2).sum() / d


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles) -> torch.Tensor:
    """pred:(...,Q) target:(...)"""
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    diff = target.unsqueeze(-1) - pred
    return torch.maximum(q * diff, (q - 1) * diff).mean()


def gaussian_nll(mean, log_std, target) -> torch.Tensor:
    return (log_std + 0.5 * ((target - mean) / log_std.exp()) ** 2).mean()


def sigreg_loss(z: torch.Tensor, n_proj: int = 256, n_t: int = 17, t_max: float = 5.0) -> torch.Tensor:
    """Sketched isotropic-Gaussian regularizer (LeJEPA/LeWM-style): compare the empirical
    characteristic function of random 1-D projections with that of N(0,1) (Epps-Pulley)."""
    z = _flat(z, 1024)
    d = z.shape[-1]
    a = torch.randn(d, n_proj, device=z.device)
    a = a / a.norm(dim=0, keepdim=True)
    t = torch.linspace(-t_max, t_max, n_t, device=z.device)
    xt = (z @ a).unsqueeze(-1) * t                              # (n,K,T)
    err = (xt.cos().mean(0) - torch.exp(-t ** 2 / 2)) ** 2 + xt.sin().mean(0) ** 2
    return torch.trapezoid(err * torch.exp(-t ** 2 / 2), t, dim=-1).mean()


def symlog(x: torch.Tensor) -> torch.Tensor:
    """DreamerV3 symmetric log: scale-free regression targets for rewards/values."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor, limit: float = 8.0) -> torch.Tensor:
    """Inverse of symlog. The clamp (|x|<=8 -> |y|<~3000) stops bootstrapped value targets from
    feeding back on themselves exponentially (this diverged to NaN without it)."""
    x = x.clamp(-limit, limit)
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


def lambda_returns(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float):
    """rewards r_0..r_{H-1}, values V_h = value of the state after step h (both (B,H), real units).
    Returns targets R_h for V_h, h=0..H-1 (R_{H-1}=V_{H-1}), and the ctx-state target."""
    H = rewards.shape[1]
    R = [None] * H
    R[H - 1] = values[:, H - 1]
    for h in reversed(range(H - 1)):
        R[h] = rewards[:, h + 1] + gamma * ((1 - lam) * values[:, h + 1] + lam * R[h + 1])
    R = torch.stack(R, 1)
    ctx = rewards[:, 0] + gamma * ((1 - lam) * values[:, 0] + lam * R[:, 0])
    return R, ctx
