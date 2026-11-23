"""Configuration + size presets for Zemlya."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Tuple


@dataclass
class ZemlyaConfig:
    # ---- data geometry -------------------------------------------------
    patch_len: int = 32            # time steps per patch (token)
    context_len: int = 512         # max context steps (multiple of patch_len)
    channels: int = 1              # observed series / state channels
    action_dim: int = 1            # continuous action vector size (one-hot discrete actions)
    max_horizon_patches: int = 8   # how many future patches the predictor can imagine
    action_low: float = -1.0
    action_high: float = 1.0

    # ---- context / target encoder (causal, TimesFM-style decoder-only) --
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4            # grouped-query attention
    ffn_mult: float = 4.0          # SwiGLU hidden = 2/3 * ffn_mult * d
    rope_base: float = 10000.0
    variate_every: int = 2         # cross-series attention after every k temporal blocks (channels>1)

    # ---- latent predictor (narrow, bidirectional, action-conditioned) ---
    pred_d_model: int = 384
    pred_layers: int = 4
    pred_heads: int = 6
    pred_kv_heads: int = 3

    # ---- heads ---------------------------------------------------------
    quantiles: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    head_hidden: int = 256
    critic_ensemble: int = 3       # reward/value ensemble -> epistemic uncertainty (pessimism)
    lam: float = 0.95              # lambda-return mixing for the value target (DreamerV3-style)
    return_dropout: float = 0.2    # drop return-conditioning of the policy this often (null token)
    decode_detach: bool = True     # heads read stop-grad latents -> JEPA stays pure

    # ---- training ------------------------------------------------------
    ema_start: float = 0.996
    ema_end: float = 1.0
    w_latent: float = 1.0
    w_var: float = 1.0
    w_cov: float = 0.04
    regularizer: str = "vicreg"    # "vicreg" | "sigreg" (isotropic-Gaussian latents, LeWM/LeJEPA style)
    w_sigreg: float = 1.0
    w_forecast: float = 1.0
    w_reward: float = 1.0
    w_value: float = 1.0
    w_policy: float = 1.0
    gamma: float = 0.97            # planning discount

    # ------------------------------------------------------------------
    def __post_init__(self):
        self.quantiles = tuple(self.quantiles)
        assert self.regularizer in ("vicreg", "sigreg")
        assert self.context_len % self.patch_len == 0, "context_len must be a multiple of patch_len"
        assert self.d_model % self.n_heads == 0 and self.n_heads % self.n_kv_heads == 0
        assert self.pred_d_model % self.pred_heads == 0 and self.pred_heads % self.pred_kv_heads == 0
        assert (self.d_model // self.n_heads) % 2 == 0 and (self.pred_d_model // self.pred_heads) % 2 == 0

    @property
    def n_context_patches(self) -> int:
        return self.context_len // self.patch_len

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ZemlyaConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ZemlyaConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def preset(cls, name: str, **overrides) -> "ZemlyaConfig":
        if name not in PRESETS:
            raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
        return cls.from_dict({**PRESETS[name], **overrides})


# Deployable (encoder + predictor + heads) sizes. See scripts/count_params.py.
PRESETS = {
    # ~tens of MB: microcontrollers / edge / unit tests
    "nano":  dict(d_model=256,  n_layers=6,  n_heads=4,  n_kv_heads=2, pred_d_model=192, pred_layers=3, pred_heads=4,  pred_kv_heads=2, head_hidden=128),
    "micro": dict(d_model=512,  n_layers=8,  n_heads=8,  n_kv_heads=4, pred_d_model=384, pred_layers=4, pred_heads=6,  pred_kv_heads=3, head_hidden=256),
    "small": dict(d_model=768,  n_layers=12, n_heads=12, n_kv_heads=4, pred_d_model=512, pred_layers=6, pred_heads=8,  pred_kv_heads=4, head_hidden=384),
    "base":  dict(d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4, pred_d_model=640, pred_layers=8, pred_heads=10, pred_kv_heads=5, head_hidden=512),
    # upper bound of the Zemlya budget (<500M deployable)
    "max":   dict(d_model=1280, n_layers=20, n_heads=20, n_kv_heads=4, pred_d_model=768, pred_layers=8, pred_heads=12, pred_kv_heads=4, head_hidden=640),
}
