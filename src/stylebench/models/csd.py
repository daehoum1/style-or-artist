"""CSD (Somepalli et al., ECCV 2024) style descriptor: CLIP ViT-L/14 visual backbone + style projection.
Official weights tomg-group-umd/CSD-ViT-L (pytorch_model.bin, keys backbone.* / last_layer_style /
last_layer_content). The backbone is architecturally identical to open_clip's OpenAI ViT-L-14 visual tower,
so weights are loaded into that module (strict, every backbone key must match)."""
import open_clip
import torch
from huggingface_hub import hf_hub_download

from .base import Encoder


def build_csd(name, cfg, device):
    model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None)
    vis = model.visual
    import numpy as np
    # weights_only stays True; only numpy scalar/dtype (stored training metadata) are allow-listed
    import argparse
    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype("float64")),
                                           argparse.Namespace]):
        sd = torch.load(hf_hub_download(cfg["pretrained"], "pytorch_model.bin"), map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    proj = sd["last_layer_style"]
    bb["proj"] = proj
    missing, unexpected = vis.load_state_dict(bb, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"CSD weight mismatch: missing={missing} unexpected={unexpected}")
    vis = vis.to(device)
    # CSD preprocessing: CLIP normalization, 224 bicubic resize + center crop (same as open_clip's default)
    return Encoder(name, vis, pre, lambda m, x: m(x), {"library": f"open_clip {open_clip.__version__}",
                                                        "input_resolution": 224})
