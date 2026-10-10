"""Dataset adapter interface.

Every adapter returns a raw metadata table with the canonical columns

    image_id  : str, unique and stable within the dataset
    path      : absolute image path
    style     : normalized style label or <NA>
    artist    : normalized artist label or <NA>
    genre     : normalized genre label or <NA>   (always <NA> when the dataset has no genre metadata)

plus any dataset-specific raw columns prefixed with ``raw_``.
"""
from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod

import pandas as pd

CANONICAL = ["image_id", "path", "style", "artist", "genre"]


class DatasetUnavailable(RuntimeError):
    """Raised when required external files are missing. The message says what is needed."""


def normalize_label(s) -> object:
    """Deterministic, non-fuzzy normalization: NFKC, '_'/'-' -> space, collapse whitespace, lowercase.
    Labels that differ after this normalization are never merged."""
    if s is None or (isinstance(s, float) and pd.isna(s)) or s is pd.NA:
        return pd.NA
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = re.sub(r"[_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s if s and s not in {"nan", "none", "unknown", "unknown genre", "unknown artist"} else pd.NA


class DatasetAdapter(ABC):
    has_genre: bool = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg["name"]

    @abstractmethod
    def required_files(self) -> list[str]:
        ...

    def check_available(self) -> None:
        from pathlib import Path
        missing = [f for f in self.required_files() if not Path(f).exists()]
        if missing:
            raise DatasetUnavailable(f"{self.name}: missing required files: {missing}")

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        ...

    def load(self) -> pd.DataFrame:
        self.check_available()
        df = self.load_raw()
        for c in CANONICAL:
            assert c in df.columns, f"adapter {self.name} did not produce column {c}"
        assert df.image_id.is_unique, "image_id must be unique"
        for c in ["style", "artist", "genre"]:
            df[c] = df[c].astype("string")
        return df.reset_index(drop=True)


def get_adapter(cfg: dict) -> DatasetAdapter:
    from .multitask_painting import MultitaskPaintingAdapter
    from .wikiart import WikiArtAdapter
    return {"wikiart": WikiArtAdapter, "multitask_painting": MultitaskPaintingAdapter}[cfg["adapter"]](cfg)
