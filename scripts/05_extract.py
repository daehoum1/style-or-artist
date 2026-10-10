"""Extract embeddings for all cleaned images. Usage: 05_extract.py <dataset> <encoder> [<encoder> ...]"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS  # noqa: E402
from stylebench.features.extract import extract  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("encoders", nargs="+")
ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N images (separate dataset tag)")
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
meta = pd.read_parquet(ARTIFACTS / "metadata" / a.dataset / "cleaned.parquet")[["image_id", "path"]]
tag = a.dataset
if a.limit:
    meta = meta.iloc[: a.limit]
    tag = f"{a.dataset}_smoke{a.limit}"
for e in a.encoders:
    extract(tag, e, meta, a.bs, a.workers)
