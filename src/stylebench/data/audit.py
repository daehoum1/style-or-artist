"""Dataset audit: counts, missingness, imbalance, label co-occurrence."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dist(s: pd.Series) -> dict:
    s = s.astype(float)
    return {"min": float(s.min()), "median": float(s.median()), "mean": round(float(s.mean()), 2),
            "max": float(s.max())} if len(s) else {}


def imbalance(counts: pd.Series) -> dict:
    c = counts.values.astype(float)
    p = c / c.sum()
    ent = -(p * np.log(p)).sum()
    return {"n_classes": int(len(c)), "max_min_ratio": round(float(c.max() / c.min()), 2),
            "gini": round(float(np.abs(np.subtract.outer(c, c)).sum() / (2 * len(c) ** 2 * c.mean())), 3),
            "normalized_entropy": round(float(ent / np.log(len(c))), 3) if len(c) > 1 else 1.0}


def audit(df: pd.DataFrame, has_genre: bool = True) -> dict:
    st, ar, ge = df["style"], df["artist"], df["genre"]
    out = {"n_images": int(len(df)),
           "missing": {c: int(df[c].isna().sum()) for c in ["style", "artist", "genre"]},
           "n_styles": int(st.nunique()), "n_artists": int(ar.nunique()),
           "n_genres": int(ge.nunique()) if has_genre else None}
    s = df.dropna(subset=["style"])
    out["images_per_style"] = _dist(s["style"].value_counts())
    out["style_imbalance"] = imbalance(s["style"].value_counts())
    sa = df.dropna(subset=["style", "artist"])
    out["artists_per_style"] = _dist(sa.groupby("style")["artist"].nunique())
    out["styles_per_artist"] = _dist(sa.groupby("artist")["style"].nunique())
    out["images_per_artist"] = _dist(sa["artist"].value_counts())
    out["share_artists_single_style"] = round(float((sa.groupby("artist")["style"].nunique() == 1).mean()), 3)
    if has_genre:
        sg = df.dropna(subset=["style", "genre"])
        out["images_per_genre"] = _dist(sg["genre"].value_counts())
        out["genres_per_style"] = _dist(sg.groupby("style")["genre"].nunique())
        out["genre_imbalance"] = imbalance(sg["genre"].value_counts())
        full = df.dropna(subset=["style", "artist", "genre"])
        out["n_images_all_three_labels"] = int(len(full))
    return out


def per_style_table(df: pd.DataFrame, has_genre: bool = True) -> pd.DataFrame:
    s = df.dropna(subset=["style"])
    t = s.groupby("style").agg(n_images=("image_id", "size"), n_artists=("artist", "nunique"),
                               n_missing_artist=("artist", lambda x: int(x.isna().sum())))
    if has_genre:
        t["n_genres"] = s.groupby("style")["genre"].nunique()
        t["n_missing_genre"] = s.groupby("style")["genre"].apply(lambda x: int(x.isna().sum()))
        t["top_genre_share"] = s.dropna(subset=["genre"]).groupby("style")["genre"].agg(
            lambda x: round(x.value_counts(normalize=True).iloc[0], 3))
    return t.sort_values("n_images", ascending=False)
