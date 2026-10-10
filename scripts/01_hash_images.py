"""Compute SHA-256 + pHash for every image of a dataset. Output: artifacts/metadata/<ds>/hashes.parquet"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, dataset_config  # noqa: E402
from stylebench.data.base import get_adapter  # noqa: E402
from stylebench.data.hashing import compute_hashes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--workers", type=int, default=20)
a = ap.parse_args()
out = ARTIFACTS / "metadata" / a.dataset
out.mkdir(parents=True, exist_ok=True)
raw = get_adapter(dataset_config(a.dataset)).load()
t = time.time()
h = compute_hashes(raw.path.tolist(), a.workers)
h.insert(0, "image_id", raw.image_id.values)
h.to_parquet(out / "hashes.parquet")
print(f"{a.dataset}: hashed {len(h)} images in {time.time()-t:.0f}s; unreadable={int((~h.readable).sum())}")
