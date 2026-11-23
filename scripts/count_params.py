"""Print parameter budgets for every preset (built on the meta device: no memory used)."""
import torch

from zemlya import PRESETS, Zemlya, ZemlyaConfig

for name in PRESETS:
    with torch.device("meta"):
        c = Zemlya(ZemlyaConfig.preset(name)).param_counts()
    print(f"{name:6s} deployable {c['deployable']/1e6:7.1f}M | "
          f"+EMA target for training {c['total_in_training']/1e6:7.1f}M | "
          f"enc {c['encoder']/1e6:.1f}M pred {c['predictor']/1e6:.1f}M")
