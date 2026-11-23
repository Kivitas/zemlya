"""Datasets. `SyntheticControlDataset` is a self-contained regulation task for smoke tests;
`ArrayDataset` wraps your own logged (obs, action, reward) trajectories."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticControlDataset(Dataset):
    """Noisy seasonal linear system  x' = a*x + b*u + season + noise ; reward = -x^2 - 0.05*u^2.
    A (partially random) feedback behavior policy produces the logged actions."""

    def __init__(self, size: int, context_len: int, horizon_patches: int, patch_len: int,
                 channels: int = 1, action_dim: int = 1, seed: int = 0):
        self.size, self.T, self.H, self.P = size, context_len, horizon_patches, patch_len
        self.C, self.A, self.seed = channels, action_dim, seed

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        T, H, P, C, A = self.T, self.H, self.P, self.C, self.A
        total = T + H * P
        a = rng.uniform(0.9, 0.99, size=C)
        b = rng.uniform(0.01, 0.04, size=(A, C))
        amp, per, ph = rng.uniform(0.1, 0.6, C), rng.uniform(16, 96, C), rng.uniform(0, 6.28, C)
        gain, explore = rng.uniform(0.3, 1.2), rng.uniform(0.0, 0.5)
        x = np.zeros((total, C), dtype=np.float32)
        x[0] = rng.normal(0, 1, C)
        acts = np.zeros((total // P + 1, A), dtype=np.float32)
        for t in range(total - 1):
            if t % P == 0:
                u = -gain * x[t].mean() * np.ones(A) + explore * rng.normal(0, 1, A)
                acts[t // P] = np.clip(u, -1, 1)
            u = acts[t // P]
            season = amp * np.sin(2 * np.pi * t / per + ph)
            x[t + 1] = a * x[t] + u @ b + 0.05 * season + 0.03 * rng.normal(0, 1, C)
        n_ctx = T // P
        fut_acts = acts[n_ctx:n_ctx + H]                                  # (H,A)
        fut = x[T:]
        r = np.array([
            -(fut[h * P:(h + 1) * P] ** 2).mean() - 0.05 * (fut_acts[h] ** 2).mean() for h in range(H)
        ], dtype=np.float32)
        return {
            "context": torch.from_numpy(x[:T]),
            "future": torch.from_numpy(fut),
            "actions": torch.from_numpy(fut_acts.copy()),
            "rewards": torch.from_numpy(r),
        }


class ArrayDataset(Dataset):
    """context:(N,T,C) future:(N,H*P,C) [actions:(N,H,A)] [rewards:(N,H)] [context_mask:(N,T)]"""

    KEYS = ("context", "future", "actions", "rewards", "context_mask")

    def __init__(self, **arrays):
        self.d = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in arrays.items() if k in self.KEYS}
        assert "context" in self.d and "future" in self.d

    @classmethod
    def from_npz(cls, path: str):
        z = np.load(path)
        return cls(**{k: z[k] for k in z.files})

    def __len__(self):
        return len(self.d["context"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.d.items()}
