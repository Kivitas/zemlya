"""Latent-space CEM planner. One encoder pass, then all candidates are scored by the (small)
predictor in a single batched call -- no environment or decoder in the loop."""
from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def plan(model, context: torch.Tensor, mask: Optional[torch.Tensor] = None, *,
         horizon_patches: Optional[int] = None, num_samples: int = 128, iters: int = 4,
         elite_frac: float = 0.1, risk_aversion: float = 0.0, action_cost: float = 0.0,
         min_std: float = 0.02, goal_future: Optional[torch.Tensor] = None, goal_weight: float = 1.0,
         init_plan: Optional[torch.Tensor] = None, target_return=None, pessimism: float = 1.0,
         generator: Optional[torch.Generator] = None):
    cfg = model.cfg
    H = horizon_patches or cfg.max_horizon_patches
    A, S = cfg.action_dim, num_samples
    lo, hi = cfg.action_low, cfg.action_high

    z_ctx, _ = model.encode(context, mask)
    B = z_ctx.shape[0]
    mean, log_std = model.policy_head(z_ctx[:, -1], H, target_return)               # amortized warm start
    if init_plan is not None:                                          # MPC warm start (shifted last plan)
        mean = init_plan[:, :H].to(mean)
    mean = mean.clamp(lo, hi)
    goal = None
    if goal_future is not None:
        goal = model.encode_goal(context, goal_future, mask)[:, :H].repeat_interleave(num_samples, dim=0)
        H = min(H, goal.shape[1])
        mean, log_std = mean[:, :H], log_std[:, :H]
    std = log_std.exp().clamp(min=min_std)
    z_rep = z_ctx.repeat_interleave(S, dim=0)

    best_score = torch.full((B,), float("-inf"), device=context.device)
    best_plan = mean.clone()
    k = max(2, int(S * elite_frac))
    for _ in range(iters):
        noise = torch.randn(B, S, H, A, device=context.device, generator=generator)
        cand = (mean[:, None] + std[:, None] * noise).clamp(lo, hi)
        cand[:, 0] = mean                                              # always re-score the mean
        score = model.score(z_rep, cand.reshape(B * S, H, A), risk_aversion, action_cost, goal, goal_weight, pessimism).view(B, S)
        top = score.topk(k, dim=1).indices
        elite = torch.gather(cand, 1, top[:, :, None, None].expand(B, k, H, A))
        mean, std = elite.mean(1), elite.std(1).clamp(min=min_std)
        b_s, b_i = score.max(1)
        improved = b_s > best_score
        best_score = torch.where(improved, b_s, best_score)
        best_plan = torch.where(improved[:, None, None], cand[torch.arange(B), b_i], best_plan)
    _, unc = model.score(z_ctx, best_plan, pessimism=pessimism, return_uncertainty=True)
    return {"action": best_plan[:, 0], "plan": best_plan, "score": best_score, "uncertainty": unc, "mode": "plan"}
