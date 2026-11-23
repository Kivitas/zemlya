"""Zemlya: JEPA-style latent world model + decision maker for small models.

    context --> [online causal encoder] --> z_ctx ----+
                                                      +--> [action-conditioned predictor] --> z_hat(future)
    candidate actions -------------------------------+                   |
                                                                         +-- latent loss <-- z_tgt <-- [EMA target encoder] <-- future
    z_hat --> forecast head (quantiles) | reward head | (+ policy head on z_ctx[:, -1])
    decision = plan(): CEM over action sequences scored entirely in latent space.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

from . import losses as L
from .config import ZemlyaConfig
from .encoder import ContextEncoder, Stats, normalize, patchify
from .heads import CriticHead, ForecastHead, PolicyHead
from .predictor import LatentPredictor


class Zemlya(nn.Module):
    def __init__(self, cfg: ZemlyaConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ContextEncoder(cfg)
        self.target_encoder: Optional[ContextEncoder] = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = LatentPredictor(cfg)
        self.forecast_head = ForecastHead(cfg)
        self.critic = CriticHead(cfg)
        self.policy_head = PolicyHead(cfg)

    # ------------------------------------------------------------------ encode / imagine
    def encode(self, context: torch.Tensor, mask: Optional[torch.Tensor] = None):
        assert context.shape[1] % self.cfg.patch_len == 0, "context length must be a multiple of patch_len"
        x, stats, mask = normalize(context, mask)
        z = self.encoder(patchify(x, mask, self.cfg.patch_len), stats)
        return z, stats

    def imagine(self, z_ctx: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.predictor(z_ctx, actions)

    # ------------------------------------------------------------------ training
    def forward(self, batch: Dict[str, torch.Tensor]):
        cfg, P = self.cfg, self.cfg.patch_len
        ctx, fut = batch["context"], batch["future"]
        mask = batch.get("context_mask")
        B, H = ctx.shape[0], fut.shape[1] // P
        actions = batch.get("actions")
        if actions is None:
            actions = ctx.new_zeros(B, H, cfg.action_dim)
        rewards = batch.get("rewards")

        x_ctx, stats, mask = normalize(ctx, mask)
        x_fut = (fut - stats.mean) / stats.std
        z_ctx = self.encoder(patchify(x_ctx, mask, P), stats)
        N = z_ctx.shape[1]

        with torch.no_grad():
            full = torch.cat([x_ctx, x_fut], dim=1)
            fmask = torch.cat([mask, mask.new_ones(B, fut.shape[1])], dim=1)
            z_tgt = self.target_encoder(patchify(full, fmask, P), stats)[:, N:N + H]

        z_hat = self.predictor(z_ctx, actions)

        m: Dict[str, torch.Tensor] = {}
        m["latent"] = L.latent_loss(z_hat, z_tgt)
        loss = cfg.w_latent * m["latent"]
        if cfg.regularizer == "sigreg":
            m["sigreg"] = L.sigreg_loss(z_ctx) + L.sigreg_loss(z_hat)
            loss = loss + cfg.w_sigreg * m["sigreg"]
        else:
            m["var"] = L.variance_loss(z_ctx) + L.variance_loss(z_hat)
            m["cov"] = L.covariance_loss(z_ctx) + L.covariance_loss(z_hat)
            loss = loss + cfg.w_var * m["var"] + cfg.w_cov * m["cov"]

        zh = z_hat.detach() if cfg.decode_detach else z_hat
        zl = z_ctx[:, -1].detach() if cfg.decode_detach else z_ctx[:, -1]

        q = self.forecast_head(zh)
        tgt = x_fut.reshape(B, H, P, cfg.channels)
        m["forecast"] = L.pinball_loss(q, tgt, cfg.quantiles)
        loss = loss + cfg.w_forecast * m["forecast"]

        ret = None
        if rewards is not None:
            disc = cfg.gamma ** torch.arange(H, device=rewards.device, dtype=rewards.dtype)
            ret = (rewards * disc).sum(1)                                   # hindsight discounted return
            r_pred = self.critic.reward(zh, actions)                       # (K,B,H) symlog
            m["reward"] = torch.nn.functional.mse_loss(r_pred, L.symlog(rewards).expand_as(r_pred))
            with torch.no_grad():                                          # lambda-return value targets
                v_all = L.symexp(self.critic.value(torch.cat([zl[:, None], zh], 1)).mean(0))   # (B,1+H)
                R, R_ctx = L.lambda_returns(rewards, v_all[:, 1:], cfg.gamma, cfg.lam)
            v_pred = self.critic.value(torch.cat([zl[:, None], zh], 1))     # (K,B,1+H)
            v_tgt = L.symlog(torch.cat([R_ctx[:, None], R], 1))[None].expand_as(v_pred)
            m["value"] = torch.nn.functional.mse_loss(v_pred[..., :-1], v_tgt[..., :-1])   # last has no signal
            loss = loss + cfg.w_reward * m["reward"] + cfg.w_value * m["value"]
        if "actions" in batch:
            keep = None
            if ret is not None:
                keep = torch.rand(B, device=ctx.device) >= cfg.return_dropout
            mean, log_std = self.policy_head(zl, H, ret, keep)
            m["policy"] = L.gaussian_nll(mean, log_std, actions)
            loss = loss + cfg.w_policy * m["policy"]

        m["loss"] = loss
        return loss, m

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for pt, po in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(momentum).add_(po.detach(), alpha=1.0 - momentum)

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def forecast(self, context: torch.Tensor, horizon_steps: int,
                 actions: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        """Quantile forecast in original units. Returns (B, horizon_steps, C, Q)."""
        cfg = self.cfg
        H = -(-horizon_steps // cfg.patch_len)
        z_ctx, stats = self.encode(context, mask)
        if actions is None:
            actions = context.new_zeros(context.shape[0], H, cfg.action_dim)
        q = self.forecast_head(self.imagine(z_ctx, actions))           # (B,H,P,C,Q)
        q = q.reshape(q.shape[0], H * cfg.patch_len, cfg.channels, -1)
        q = q * stats.std.unsqueeze(-1) + stats.mean.unsqueeze(-1)
        return q[:, :horizon_steps]

    # ------------------------------------------------------------------ decision
    @torch.no_grad()
    def encode_goal(self, context: torch.Tensor, goal_future: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Latents of a *desired* future trajectory (B,H*P,C) -- e.g. a constant setpoint.
        Built exactly like training targets (context prefix + future); uses the EMA target
        encoder if present, else the online encoder."""
        P = self.cfg.patch_len
        x_ctx, stats, mask = normalize(context, mask)
        x_goal = (goal_future - stats.mean) / stats.std
        B, H = context.shape[0], goal_future.shape[1] // P
        full = torch.cat([x_ctx, x_goal], 1)
        fmask = torch.cat([mask, mask.new_ones(B, goal_future.shape[1])], 1)
        enc = self.target_encoder if self.target_encoder is not None else self.encoder
        N = x_ctx.shape[1] // P
        return enc(patchify(full, fmask, P), stats)[:, N:N + H]

    def score(self, z_ctx: torch.Tensor, actions: torch.Tensor,
              risk_aversion: float = 0.0, action_cost: float = 0.0,
              goal_latents: Optional[torch.Tensor] = None, goal_weight: float = 1.0,
              pessimism: float = 1.0, return_uncertainty: bool = False):
        """Score candidate action sequences purely in latent space (MuZero-style: imagined
        discounted rewards + terminal value). (B,N,D),(B,H,A) -> (B,)
          pessimism      penalizes critic-ensemble disagreement (epistemic uncertainty)
          risk_aversion  penalizes forecast quantile spread (aleatoric uncertainty)
          goal_latents   (B,H,D) adds a JEPA-WM-style latent-distance tracking cost"""
        z_hat = self.imagine(z_ctx, actions)
        H = actions.shape[1]
        disc = self.cfg.gamma ** torch.arange(H, device=actions.device, dtype=z_hat.dtype)
        r = L.symexp(self.critic.reward(z_hat, actions))                    # (K,B,H)
        v = L.symexp(self.critic.value(z_hat[:, -1:]))[..., 0]              # (K,B)
        ret = (r * disc).sum(-1) + self.cfg.gamma ** H * v                  # (K,B)
        unc = ret.std(0, unbiased=False)
        total = ret.mean(0) - pessimism * unc
        if risk_aversion > 0:
            q = self.forecast_head(z_hat)
            total = total - risk_aversion * (q[..., -1] - q[..., 0]).mean(dim=(1, 2, 3))
        if goal_latents is not None:
            ph = torch.nn.functional.layer_norm(z_hat.float(), z_hat.shape[-1:])
            gl = torch.nn.functional.layer_norm(goal_latents.float(), z_hat.shape[-1:])
            total = total - goal_weight * (ph - gl).pow(2).mean(dim=(1, 2))
        if action_cost > 0:
            total = total - action_cost * actions.pow(2).mean(dim=(1, 2))
        return (total, unc) if return_uncertainty else total

    # ---- three operating modes (cf. WHAM: world model / policy / both) --------------------
    @torch.no_grad()
    def act(self, context: torch.Tensor, mask: Optional[torch.Tensor] = None, *,
            horizon_patches: Optional[int] = None, target_return=None, pessimism: float = 1.0):
        """'System 1': amortized policy only. One encoder pass + one predictor pass (for uncertainty)."""
        H = horizon_patches or self.cfg.max_horizon_patches
        z_ctx, _ = self.encode(context, mask)
        mean, _ = self.policy_head(z_ctx[:, -1], H, target_return)
        plan = mean.clamp(self.cfg.action_low, self.cfg.action_high)
        score, unc = self.score(z_ctx, plan, pessimism=pessimism, return_uncertainty=True)
        return {"action": plan[:, 0], "plan": plan, "score": score, "uncertainty": unc, "mode": "policy"}

    @torch.no_grad()
    def decide(self, context: torch.Tensor, mask: Optional[torch.Tensor] = None, *,
               mode: str = "plan", uncertainty_threshold: Optional[float] = None, **kw):
        """mode='policy' (fast, amortized) | 'plan' (CEM search, 'System 2') | 'auto'
        (policy first; search only for rows whose critic-ensemble uncertainty exceeds the threshold
        -- adaptive compute, ideal for small models)."""
        from .planner import plan
        if mode == "plan":
            return plan(self, context, mask, **kw)
        act_kw = {k: kw[k] for k in ("horizon_patches", "target_return", "pessimism") if k in kw}
        out = self.act(context, mask, **act_kw)
        if mode == "policy":
            return out
        assert mode == "auto" and uncertainty_threshold is not None
        need = out["uncertainty"] > uncertainty_threshold
        out["searched"] = need
        if need.any():
            idx = need.nonzero().squeeze(-1)
            sub = dict(kw)
            sub["init_plan"] = out["plan"][idx]
            if sub.get("goal_future") is not None:
                sub["goal_future"] = sub["goal_future"][idx]
            tr = sub.get("target_return")
            if torch.is_tensor(tr) and tr.ndim > 0:
                sub["target_return"] = tr[idx]
            res = plan(self, context[idx], None if mask is None else mask[idx], **sub)
            H = res["plan"].shape[1]
            out["plan"] = out["plan"].clone()
            out["plan"][idx, :H] = res["plan"]
            out["action"] = out["plan"][:, 0]
            out["score"] = out["score"].clone()
            out["score"][idx] = res["score"]
            out["uncertainty"] = out["uncertainty"].clone()
            out["uncertainty"][idx] = res["uncertainty"]
        out["mode"] = "auto"
        return out

    # ------------------------------------------------------------------ utils
    def strip_for_inference(self) -> "Zemlya":
        """Drop the EMA target encoder (only needed for training)."""
        self.target_encoder = None
        return self

    def param_counts(self) -> Dict[str, int]:
        def n(m): return sum(p.numel() for p in m.parameters())
        out = {"encoder": n(self.encoder), "predictor": n(self.predictor),
               "forecast_head": n(self.forecast_head), "critic": n(self.critic),
               "policy_head": n(self.policy_head)}
        out["deployable"] = sum(out.values())
        out["target_encoder(train only)"] = n(self.target_encoder) if self.target_encoder is not None else 0
        out["total_in_training"] = out["deployable"] + out["target_encoder(train only)"]
        return out
