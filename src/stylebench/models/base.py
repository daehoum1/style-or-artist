"""Frozen encoder interface: build_encoder(name) -> Encoder with .transform (PIL -> tensor), .encode(batch)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..config import load_yaml


@dataclass
class Encoder:
    name: str
    model: torch.nn.Module
    transform: object
    forward: object                    # callable(model, x) -> (B, D) tensor
    info: dict = field(default_factory=dict)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(self.model, x).float()


def encoder_configs() -> dict:
    return load_yaml("configs/encoders/encoders.yaml")


def build_encoder(name: str, device: str) -> Encoder:
    cfg = encoder_configs()[name]
    lib = cfg["lib"]
    if lib == "open_clip":
        from .vl import build_open_clip
        enc = build_open_clip(name, cfg, device)
    elif lib == "timm":
        from .timm_models import build_timm
        enc = build_timm(name, cfg, device)
    elif lib == "aladin":
        from .aladin import build_aladin
        enc = build_aladin(name, cfg, device)
    elif lib == "csd":
        from .csd import build_csd
        enc = build_csd(name, cfg, device)
    else:
        raise ValueError(lib)
    enc.model.eval().requires_grad_(False)
    enc.info.update({k: v for k, v in cfg.items()})
    enc.info["n_params"] = int(sum(p.numel() for p in enc.model.parameters()))
    enc.info["transform"] = repr(enc.transform)
    return enc
