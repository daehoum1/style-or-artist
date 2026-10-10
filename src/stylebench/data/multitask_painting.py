"""MultitaskPainting100k adapter (original train+test ground truth pooled)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import DatasetAdapter, normalize_label


class MultitaskPaintingAdapter(DatasetAdapter):
    def required_files(self):
        root = Path(self.cfg["root"])
        return [str(root / f) for f in self.cfg["label_files"]] + [str(root / self.cfg["image_dir"])]

    def load_raw(self) -> pd.DataFrame:
        root = Path(self.cfg["root"])
        parts = []
        for f in self.cfg["label_files"]:
            d = pd.read_csv(root / f)
            d["raw_original_subset"] = "test" if "test" in f else "train"
            parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        dup = d.filename.duplicated(keep=False)
        if dup.any():
            raise ValueError(f"MP100k: {dup.sum()} filenames listed more than once")
        img = root / self.cfg["image_dir"]
        exists = d.filename.map(lambda f: (img / f).exists())
        d = d[exists].copy()
        d.attrs["missing_image_files"] = int((~exists).sum())
        return pd.DataFrame({
            "image_id": d.filename.values,
            "path": [str(img / f) for f in d.filename],
            "style": d["style"].map(normalize_label).values,
            "artist": d["artist"].map(normalize_label).values,
            "genre": d["genre"].map(normalize_label).values,
            "raw_style": d["style"].values, "raw_artist": d["artist"].values,
            "raw_genre": d["genre"].values,
            "raw_original_subset": d["raw_original_subset"].values,
        })
