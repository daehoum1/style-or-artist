import json

import numpy as np
import pandas as pd

from ..config import ARTIFACTS


def load_embeddings(dataset: str, encoder: str, normalize: bool = True):
    d = ARTIFACTS / "embeddings" / dataset / encoder
    E = np.load(d / "embeddings.npy").astype(np.float32)
    meta = pd.read_parquet(d / "metadata.parquet")
    assert len(meta) == len(E) and meta.row.tolist() == list(range(len(E)))
    if normalize:
        E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E, dict(zip(meta.image_id, meta.row)), json.load(open(d / "config.json"))


def available(dataset: str, encoder: str) -> bool:
    return (ARTIFACTS / "embeddings" / dataset / encoder / "embeddings.npy").exists()
