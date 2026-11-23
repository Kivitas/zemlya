"""Training loop:  python -m zemlya.train --preset nano --steps 300"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .config import PRESETS, ZemlyaConfig
from .data import ArrayDataset, SyntheticControlDataset
from .model import Zemlya


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="nano", choices=list(PRESETS))
    ap.add_argument("--config", default=None, help="JSON config (overrides preset)")
    ap.add_argument("--npz", default=None, help="npz with context/future[/actions/rewards]; default: synthetic")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--horizon-patches", type=int, default=4)
    ap.add_argument("--context-len", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/zemlya")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    over = {"context_len": args.context_len} if args.context_len else {}
    cfg = ZemlyaConfig.from_json(args.config) if args.config else ZemlyaConfig.preset(args.preset, **over)
    cfg.max_horizon_patches = max(cfg.max_horizon_patches, args.horizon_patches)

    ds = (ArrayDataset.from_npz(args.npz) if args.npz else
          SyntheticControlDataset(100_000, cfg.context_len, args.horizon_patches, cfg.patch_len,
                                  cfg.channels, cfg.action_dim, args.seed))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=bool(args.npz), drop_last=True, num_workers=0)

    model = Zemlya(cfg).to(args.device)
    print({k: f"{v/1e6:.1f}M" for k, v in model.param_counts().items()})
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    use_amp = args.device.startswith("cuda")

    os.makedirs(args.out, exist_ok=True)
    cfg.to_json(os.path.join(args.out, "config.json"))
    step, t0, it = 0, time.time(), iter(dl)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch = {k: v.to(args.device) for k, v in batch.items()}
        lr = args.lr * (step + 1) / args.warmup if step < args.warmup else \
            args.lr * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup)))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=use_amp):
            loss, m = model(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        mom = cfg.ema_end - (cfg.ema_end - cfg.ema_start) * (math.cos(math.pi * step / args.steps) + 1) / 2
        model.update_target(mom)
        step += 1
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:5d} lr {lr:.2e} " + " ".join(f"{k} {v.item():.4f}" for k, v in m.items())
                  + f" | {time.time()-t0:.0f}s", flush=True)
    path = os.path.join(args.out, "zemlya.pt")
    torch.save({"cfg": cfg.to_dict(), "model": model.state_dict()}, path)
    print("saved", path)
    return model


if __name__ == "__main__":
    main()
