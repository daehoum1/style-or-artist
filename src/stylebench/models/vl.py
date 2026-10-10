"""CLIP / OpenCLIP / SigLIP via open_clip. Embedding = projected image embedding (before L2 norm)."""
import open_clip

from .base import Encoder


def build_open_clip(name, cfg, device):
    if cfg["arch"].startswith("hf-hub:"):
        model, _, pre = open_clip.create_model_and_transforms(cfg["arch"])
    else:
        model, _, pre = open_clip.create_model_and_transforms(cfg["arch"], pretrained=cfg["pretrained"])
    model = model.to(device)
    return Encoder(name, model, pre, lambda m, x: m.encode_image(x),
                   {"library": f"open_clip {open_clip.__version__}",
                    "input_resolution": model.visual.image_size})
