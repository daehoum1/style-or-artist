"""timm backbones (DINOv2/DINOv3/MAE/supervised ViT). Uses the model's own pretrained data config."""
import timm
import torch

from .base import Encoder


def build_timm(name, cfg, device):
    model = timm.create_model(cfg["arch"], pretrained=True, num_classes=0).to(device)
    dc = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**dc, is_training=False)
    feat = cfg["feature"]

    def fwd(m, x):
        t = m.forward_features(x)           # (B, N, D) incl. prefix tokens, after final norm
        npt = getattr(m, "num_prefix_tokens", 1)
        if feat in ("cls_token", "cls_token_prelogits"):
            return t[:, 0]
        if feat == "mean_patch_tokens":
            return t[:, npt:].mean(1)
        raise ValueError(feat)
    return Encoder(name, model, tf, fwd, {"library": f"timm {timm.__version__}",
                                          "input_resolution": dc["input_size"], "data_config": dc})
