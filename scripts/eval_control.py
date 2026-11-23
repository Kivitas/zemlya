"""Closed-loop check on the synthetic regulation task: does deciding beat doing nothing / random?
   python scripts/eval_control.py --ckpt runs/zemlya/zemlya.pt"""
import argparse

import numpy as np
import torch

from zemlya.inference import load


class Env:
    """Same dynamics as SyntheticControlDataset, vectorized over episodes (C=1, A=1)."""

    def __init__(self, n, P, warm_steps, seed=0, shock=0.0):
        r = np.random.default_rng(seed)
        self.n, self.P, self.r = n, P, r
        self.a, self.b = r.uniform(0.9, 0.99, n), r.uniform(0.01, 0.04, n)
        self.amp, self.per, self.ph = r.uniform(0.1, 0.6, n), r.uniform(16, 96, n), r.uniform(0, 6.28, n)
        self.t, self.x = 0, [r.normal(0, 1, n)]
        gain, expl = r.uniform(0.3, 1.2, n), r.uniform(0.0, 0.5, n)
        for k in range(warm_steps // P):                      # logged history under the behavior policy
            u = np.clip(-gain * self.x[-1] + expl * r.normal(0, 1, n), -1, 1)
            self._run(u)
        self.x[-1] = self.x[-1] + shock * r.normal(0, 1, n)   # disturbance at decision time

    def _run(self, u):
        for _ in range(self.P):
            x = self.x[-1]
            season = self.amp * np.sin(2 * np.pi * self.t / self.per + self.ph)
            self.x.append(self.a * x + u * self.b + 0.05 * season + 0.03 * self.r.normal(0, 1, self.n))
            self.t += 1

    def obs(self, T):
        return torch.tensor(np.stack(self.x[-T:], 1), dtype=torch.float32).unsqueeze(-1)

    def step(self, u):
        s = len(self.x)
        self._run(u)
        return float(np.mean(np.array(self.x[s:]) ** 2))


def rollout(model, policy, n, steps, T, P, seed, shock=0.0):
    env, cost = Env(n, P, T, seed, shock), []
    for _ in range(steps):
        u = policy(env.obs(T))
        cost.append(env.step(u))
    return float(np.mean(cost))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/zemlya/zemlya.pt")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--shock", type=float, default=2.0, help="state disturbance at decision time")
    ap.add_argument("--pessimism", type=float, default=1.0)
    ap.add_argument("--action-cost", type=float, default=0.0)
    a = ap.parse_args()
    m = load(a.ckpt)
    T, P = m.cfg.context_len, m.cfg.patch_len
    rng = np.random.default_rng(1)
    pol = {
        "zero action": lambda o: np.zeros(len(o)),
        "random": lambda o: rng.uniform(-1, 1, len(o)),
        "zemlya policy": lambda o: m.decide(o, mode="policy", horizon_patches=2)["action"][:, 0].numpy(),
        "zemlya plan": lambda o: m.decide(o, mode="plan", horizon_patches=2, num_samples=128, iters=3, pessimism=a.pessimism, action_cost=a.action_cost)["action"][:, 0].numpy(),
        "zemlya auto": lambda o: m.decide(o, mode="auto", uncertainty_threshold=0.05, horizon_patches=2,
                                          num_samples=128, iters=3, pessimism=a.pessimism, action_cost=a.action_cost)["action"][:, 0].numpy(),
    }
    for name, f in pol.items():
        print(f"{name:14s} mean x^2 (lower is better): {rollout(m, f, a.episodes, a.steps, T, P, seed=7, shock=a.shock):.4f}")
