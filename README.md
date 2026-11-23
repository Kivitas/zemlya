# Zemlya

A **JEPA-style latent decision & prediction architecture** for small models (**< 500M parameters**).
Architecture only: no pretrained weights are shipped. It is *inspired by* Google's TimesFM
(patched decoder-only transformer, quantile outputs) -- it is not a TimesFM port.

### Design lineage (what was borrowed from where)
| idea | from | in Zemlya |
|---|---|---|
| Latent prediction, EMA target, anti-collapse | JEPA family ([V-JEPA 2](https://arxiv.org/html/2506.09985v1), [LeWorldModel](https://arxiv.org/html/2603.19312)) | `ContextEncoder` + `LatentPredictor`; `regularizer = vicreg \| sigreg` |
| Goal-conditioned latent planning, MPC warm start | [JEPA-WM study](https://arxiv.org/html/2512.24497v2), DINO-WM | `decide(goal_future=..., init_plan=...)` |
| Patched causal transformer, cross-series attention, 9 monotone quantiles | Google [TimesFM-3](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) / [TimesFM](https://github.com/google-research/timesfm) | channel x patch grid, `VariateBlock`, `ForecastHead` |
| One model = world model *and* behavior policy | Microsoft [WHAM / Muse](https://www.microsoft.com/en-us/research/?p=1122837) | `imagine()` (world) + `act()` (policy) + `decide()` (both) |
| Symlog targets, critic trained on imagined latents, lambda-returns | Google/DeepMind [DreamerV3](https://arxiv.org/html/2301.04104v2) | `CriticHead`, `losses.symlog/lambda_returns` |
| Search with value + policy prior | DeepMind MuZero (secondary source only) | CEM warm-started by policy, terminal value in `score()` |
| Return-conditioned policy | Decision Transformer (from memory; not found in my searches) | `PolicyHead(ret=...)`, hindsight return, conditioning dropout |
| Forecaster as a component of offline decision-making | [FORL](https://arxiv.org/abs/2512.01987) | forecast head shares the latent used for planning |

Extras: critic **ensemble disagreement** = epistemic-uncertainty penalty (`pessimism`), forecast quantile spread = risk penalty
(`risk_aversion`), and **adaptive compute** (`mode="auto"`: cheap policy first, CEM search only where uncertainty is high).

## Idea

Predict in **latent space**, not in signal space, then decide by searching over imagined futures.

```
context (B,T,C) --instance-norm--> patches --> [Online causal encoder] --> z_ctx
                                                                              |
candidate actions (B,H,A) -------------------------------> [Latent predictor] --> z_hat (B,H,D)
                                                                              |         |
future --> [EMA target encoder, stop-grad] --> z_tgt  ---- latent loss -------+         +--> forecast head (quantiles)
                                                                                          +--> reward head
z_ctx[:, -1] (+ desired return) --> policy head (action prior, warm-starts planner)
decision = policy | CEM | auto; score = sum gamma^h r_h + gamma^H V(z_H) - pessimism*ensemble_std - risk*quantile_spread
```

Why this suits small models:
- **One encoder pass per decision**; candidates only touch the narrow predictor (~15-20% of params).
- **No autoregressive decoding** -- all H future latents are predicted in parallel.
- **Heads read stop-grad latents** (`decode_detach`), so the JEPA objective stays pure.
- **Anti-collapse**: EMA target + VICReg variance/covariance terms.
- **Risk-aware**: forecast quantile spread penalizes uncertain plans (`risk_aversion`).
- GQA, RoPE, RMSNorm, SwiGLU; instance norm with level/scale fed back as a token bias.

## Presets (deployable = encoder + predictor + heads)

| preset | deployable (C=1) | C=8 | + EMA copy while training (C=1) |
|-------|---:|---:|---:|
| nano  | 6.3M   | ~7.1M  | 10.8M  |
| micro | 32.6M  | ~36.3M | 56.5M  |
| small | 98.6M  | ~108.8M| 174.7M |
| base  | 222.9M | ~245M  | 401.3M |
| max   | 408.8M | ~449M  | 754.6M |

The EMA target encoder is training-only (`model.strip_for_inference()`). `python scripts/count_params.py` reprints this.

## Usage

```bash
pip install -e .
pytest -q
python -m zemlya.train --preset nano --steps 300          # synthetic control task
```

```python
import torch
from zemlya import Zemlya, ZemlyaConfig
from zemlya.inference import load

model = load("runs/zemlya/zemlya.pt")
ctx = torch.randn(1, 512, 1)                              # (B, T, C), T multiple of patch_len
q = model.forecast(ctx, horizon_steps=96)                 # (B, 96, C, n_quantiles)
d = model.decide(ctx, mode="policy", target_return=-0.5)      # fast, amortized
d = model.decide(ctx, mode="plan", horizon_patches=4, risk_aversion=0.5)
d = model.decide(ctx, mode="auto", uncertainty_threshold=0.05)  # search only when unsure
setpoint = torch.zeros(1, 4 * 32, 1)                      # desired future trajectory
d = model.decide(ctx, horizon_patches=4, goal_future=setpoint, goal_weight=1.0)
d["action"], d["plan"], d["score"]
```

Own data: `.npz` with `context (N,T,C)`, `future (N,H*P,C)`, optional `actions (N,H,A)` (one per future patch),
`rewards (N,H)`, `context_mask (N,T)` -> `--npz file.npz`. Discrete actions: one-hot them into `action_dim`.

## Layout

```
zemlya/config.py     config + presets      zemlya/predictor.py  action-conditioned latent predictor
zemlya/layers.py     RMSNorm/RoPE/GQA/..   zemlya/heads.py      forecast / reward / policy heads
zemlya/encoder.py    norm, patching, enc   zemlya/losses.py     latent, VICReg, pinball, NLL
zemlya/model.py      Zemlya (train/forecast/score)   zemlya/planner.py   CEM planner
zemlya/data.py       synthetic + array datasets      zemlya/train.py     training loop
configs/*.json       presets               tests/ scripts/
```

## Limits

Reference architecture; the synthetic task is a smoke test, not a benchmark. Horizon is capped at
`max_horizon_patches` (no autoregressive extension yet); padding is masked in normalization but not in attention.

## Evaluation so far (honest)
`python scripts/eval_control.py --ckpt runs/zemlya/zemlya.pt` runs a closed loop on the synthetic regulation task
(nano, 2500 steps, CPU, state shocked at decision time; mean x^2, lower is better):

| zero action | random | Zemlya policy | Zemlya CEM plan | auto |
|---:|---:|---:|---:|---:|
| 0.418 | 0.483 | **0.376** | 0.444 | 0.438 |

The policy mode beats doing nothing by ~10%; **the CEM planner does not yet beat the baselines** on this tiny model
and task: the world model learns the sign of the action effect but the reward/value heads are too coarse to rank
nearby plans. A first version of this benchmark was ill-posed (logged expert worse than zero action), and the value
bootstrap diverged to NaN until `symexp` was clamped. Treat the planner as unproven; try larger presets, longer
training, and real data before relying on it.

## Sources consulted
- [Microsoft WHAM/Muse](https://www.microsoft.com/en-us/research/?p=1122837), [DreamerV3](https://arxiv.org/html/2301.04104v2), [FORL](https://arxiv.org/abs/2512.01987)
- [TimesFM-3 blog](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) (the Hugging Face model card describes its architecture differently; unverified against the paper)
- [TimesFM repo](https://github.com/google-research/timesfm) (2.5: ~200M params, up to 16k context, optional quantile head, covariates)
- [LeWorldModel](https://arxiv.org/html/2603.19312), [JEPA-WM planning study](https://arxiv.org/html/2512.24497v2), [V-JEPA 2](https://arxiv.org/html/2506.09985v1)

## License

MIT, Copyright (c) 2026 kivitas. See [LICENSE](LICENSE).
