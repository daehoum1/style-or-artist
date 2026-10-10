"""Leakage probes on frozen embeddings.

global probes      : embedding -> style | artist | genre      (controlled pool, stratified 70/30 splits)
within-style probes: for each eligible style separately, embedding -> artist | genre; averaged over styles
Eligibility of classes/styles is decided from metadata only and is identical for every encoder.
Reported per probe: kNN-5 and logistic regression (wd selected on an inner 10% validation split);
balanced accuracy, macro-F1 and chance-normalized balanced accuracy (bAcc - 1/K) / (1 - 1/K).
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .classifiers import knn_predict, logreg_select_and_fit
from .metrics import confusion, metrics_from_confusion


def _cap(df, col, cap, rng):
    if not cap:
        return df
    keep = []
    for _, idx in df.groupby(col).indices.items():
        keep.append(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
    return df.iloc[np.sort(np.concatenate(keep))]


def _strat(y, frac, rng):
    m = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = min(max(int(round(frac * len(idx))), 1), len(idx) - 1)
        m[rng.choice(idx, k, replace=False)] = True
    return m


def probe_tasks(pool: pd.DataFrame, pcfg: dict, has_genre: bool) -> list[dict]:
    """Metadata-only definition of every probe task: name, style (or None), label column, row subset."""
    tasks = [{"probe": "style", "style": None, "label": "style", "rows": pool.index.to_numpy(), "cap": None}]
    vc = pool["artist"].value_counts()
    arts = vc.index[vc >= pcfg["global_artist_min_images"]]
    tasks.append({"probe": "artist", "style": None, "label": "artist",
                  "rows": pool.index[pool["artist"].isin(arts)].to_numpy(), "cap": pcfg["max_images_per_class"]})
    if has_genre:
        tasks.append({"probe": "genre", "style": None, "label": "genre", "rows": pool.index.to_numpy(), "cap": None})
    for s, sub in pool.groupby("style"):
        c = sub["artist"].value_counts()
        ok = c.index[c >= pcfg["within_style_min_images_per_artist"]]
        if len(ok) >= pcfg["within_style_min_artists"]:
            tasks.append({"probe": "within_style_artist", "style": s, "label": "artist",
                          "rows": sub.index[sub["artist"].isin(ok)].to_numpy(), "cap": pcfg["max_images_per_class"]})
        if has_genre:
            c = sub["genre"].value_counts()
            ok = c.index[c >= pcfg["within_style_min_images_per_genre"]]
            if len(ok) >= pcfg["within_style_min_genres"]:
                tasks.append({"probe": "within_style_genre", "style": s, "label": "genre",
                              "rows": sub.index[sub["genre"].isin(ok)].to_numpy(),
                              "cap": pcfg["max_images_per_class"]})
    return tasks


def run_probes(E, row, pool: pd.DataFrame, tasks: list[dict], pcfg: dict, wd_grid, max_iter=500) -> pd.DataFrame:
    out = []
    for t in tasks:
        for seed in pcfg["seeds"]:
            # deterministic across processes (Python's str hash is salted, so crc32 is used)
            rng = np.random.default_rng([seed, zlib.crc32(t["probe"].encode()), zlib.crc32(str(t["style"]).encode())])
            sub = _cap(pool.loc[t["rows"]], t["label"], t["cap"], rng)
            cls = sorted(sub[t["label"]].unique())
            y = sub[t["label"]].map({c: i for i, c in enumerate(cls)}).to_numpy()
            K = len(cls)
            X = E[sub.image_id.map(row).to_numpy()]
            te = _strat(y, pcfg["test_fraction"], rng)
            tr = ~te
            p = knn_predict(X[tr], y[tr], X[te], [5], K)[5]
            m = metrics_from_confusion(confusion(y[te], p, K))
            base = {"probe": t["probe"], "style": t["style"], "seed": seed, "n_classes": K,
                    "n_train": int(tr.sum()), "n_test": int(te.sum())}
            out.append({**base, "evaluator": "knn5", **{k: v for k, v in m.items() if k != "per_class"}})
            # logistic regression with inner validation (warm-started path, see classifiers.py)
            itr = np.flatnonzero(tr)
            iv = _strat(y[itr], 0.1, rng)
            sc = lambda yt, pp: metrics_from_confusion(confusion(yt, pp, K))["balanced_accuracy"]  # noqa: E731
            pt, bw, _ = logreg_select_and_fit(X[itr[~iv]], y[itr[~iv]], X[itr[iv]], y[itr[iv]], X[tr], y[tr],
                                              X[te], K, wd_grid, max_iter, seed, sc)
            m = metrics_from_confusion(confusion(y[te], pt, K))
            out.append({**base, "evaluator": "logreg", "hparam": bw,
                        **{k: v for k, v in m.items() if k != "per_class"}})
    return pd.DataFrame(out)
