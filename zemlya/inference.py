"""Load a checkpoint for deployment (EMA target encoder dropped)."""
from __future__ import annotations

import torch

from .config import ZemlyaConfig
from .model import Zemlya


def load(path: str, device: str = "cpu", strip: bool = True) -> Zemlya:
    ck = torch.load(path, map_location=device)
    model = Zemlya(ZemlyaConfig.from_dict(ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model.strip_for_inference() if strip else model
