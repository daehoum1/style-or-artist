"""ALADIN-ViT (Ruta et al., ICCV 2021; ViT variant from StyleBabel, ECCV 2022), official inference code and
weights from github.com/DanRuta/ALADIN (release v1.0.0, aladinVIT_bamfg_64.47.pt, trained on BAM-FG).
Preprocessing follows the repository's main.py: resize to 224x224 (no crop), normalize with mean=std=0.5.
Embedding = projected style code (the repository's AViT.encode)."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

from .base import Encoder

TP = Path(__file__).resolve().parents[3] / "third_party" / "aladin"


def build_aladin(name, cfg, device):
    sys.path.insert(0, str(TP))
    from vit_networks import VisionTransformer  # official model definition (vendored, unmodified)

    class AViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_style = VisionTransformer()

    m = AViT()
    sd = torch.load(TP / "weights" / "aladinVIT_bamfg_64.47.pt", map_location="cpu")
    m.load_state_dict(sd)  # strict
    m = m.to(device)
    tf = T.Compose([T.Lambda(lambda im: im.convert("RGB")), T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return Encoder(name, m, tf, lambda mm, x: mm.enc_style(x, just_style_code=True)[0],
                   {"library": "DanRuta/ALADIN v1.0.0 (vendored)", "input_resolution": 224})
