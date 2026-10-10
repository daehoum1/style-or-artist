"""Embedding extraction with caching, resume (chunked shards), fp16 autocast and integrity checks.
Layout: artifacts/embeddings/<dataset>/<encoder>/{embeddings.npy, metadata.parquet, config.json}"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import ARTIFACTS
from ..models.base import build_encoder

Image.MAX_IMAGE_PIXELS = None


class ImgDS(Dataset):
    def __init__(self, paths, tf, draft_px):
        self.paths, self.tf, self.draft_px = paths, tf, draft_px

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            # JPEG draft decoding to >= 2x the model's input size (deterministic, much faster on 4k scans)
            im.draft("RGB", (self.draft_px, self.draft_px))
            im = im.convert("RGB")
            # numpy (not torch) so batches travel through the worker pipe: /dev/shm is only 64 MB here
            return self.tf(im).numpy().astype(np.float16)


def extract(dataset: str, encoder: str, meta: pd.DataFrame, batch_size=256, workers=16, chunk=8192,
            device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = ARTIFACTS / "embeddings" / dataset / encoder
    final = out / "embeddings.npy"
    if final.exists():
        old = pd.read_parquet(out / "metadata.parquet")
        if old.image_id.tolist() == meta.image_id.tolist():
            print(f"[skip] {dataset}/{encoder} exists")
            return
        raise RuntimeError(f"{final} exists but for a different image list; move it away explicitly")
    (out / "shards").mkdir(parents=True, exist_ok=True)
    enc = build_encoder(encoder, device)
    res = enc.info.get("input_resolution", 224)
    res = max(res) if isinstance(res, (tuple, list)) else res
    paths = meta.path.tolist()
    t0 = time.time()
    for c0 in range(0, len(paths), chunk):
        sp = out / "shards" / f"{c0:08d}.npy"
        if sp.exists():
            continue
        dl = DataLoader(ImgDS(paths[c0:c0 + chunk], enc.transform, 2 * res), batch_size=batch_size,
                        num_workers=workers, collate_fn=np.stack, prefetch_factor=2)
        feats = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            for x in dl:
                feats.append(enc.encode(torch.from_numpy(x).to(device).float()).cpu().numpy())
        np.save(sp, np.concatenate(feats).astype(np.float32))
        print(f"{dataset}/{encoder}: {min(c0+chunk, len(paths))}/{len(paths)} {time.time()-t0:.0f}s", flush=True)
    E = np.concatenate([np.load(out / "shards" / f"{c0:08d}.npy") for c0 in range(0, len(paths), chunk)])
    # integrity checks
    assert E.shape[0] == len(meta), "row count mismatch"
    assert np.isfinite(E).all(), "non-finite embeddings"
    norms = np.linalg.norm(E, axis=1)
    assert (norms > 1e-6).all(), "zero embeddings"
    np.save(final, E)
    meta[["image_id"]].assign(row=np.arange(len(meta))).to_parquet(out / "metadata.parquet")
    info = {k: (v if isinstance(v, (int, float, str, list, dict, type(None))) else str(v))
            for k, v in enc.info.items()}
    info.update({"dataset": dataset, "encoder": encoder, "n": int(E.shape[0]), "dim": int(E.shape[1]),
                 "torch": torch.__version__, "python": platform.python_version(),
                 "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
                 "autocast": "fp16" if device == "cuda" else "fp32", "jpeg_draft_px": 2 * res,
                 "seconds": round(time.time() - t0, 1), "norm_mean": float(norms.mean())})
    json.dump(info, open(out / "config.json", "w"), indent=1, default=str)
    for s in (out / "shards").glob("*.npy"):
        s.unlink()
    (out / "shards").rmdir()
    print(f"[done] {dataset}/{encoder} {E.shape} {time.time()-t0:.0f}s")
