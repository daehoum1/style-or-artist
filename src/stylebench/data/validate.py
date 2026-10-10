"""Leakage assertions for split files. Every check raises SplitLeakageError on failure."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hashing import near_duplicate_pairs
from .splits import MF_SEEN, NEEDS_ARTIST_DISJOINT, NEEDS_GENRE_DISJOINT


class SplitLeakageError(AssertionError):
    pass


def _check(cond, msg, results, name):
    results[name] = bool(cond)
    if not cond:
        raise SplitLeakageError(msg)


def validate_split(df: pd.DataFrame, protocol: str, sup: dict, hashes: pd.DataFrame | None = None,
                   max_hamming: int = 4) -> dict:
    """Return a dict of named check -> True. Raises on the first failure."""
    r: dict = {}
    _check(df.image_id.is_unique, "image appears in more than one part", r, "image_unique")
    test = df[df.part == "test"]
    side = df[df.part != "test"]                      # train + val + selection-excluded
    val = df[df.part == "val"]
    fit = df[df.part == "train"]
    a_ov = set(side["artist"]) & set(test["artist"])
    g_ov = set(side["genre"].dropna()) & set(test["genre"].dropna())
    r["artist_overlap_count"] = len(a_ov)
    r["genre_overlap_count"] = len(g_ov)
    if protocol in MF_SEEN:
        a_seen, g_seen = MF_SEEN[protocol]
        if not a_seen:
            _check(not a_ov, f"mf artist leakage: {sorted(a_ov)[:5]}", r, "artist_disjoint_train_test")
        if not g_seen:
            _check(not g_ov, f"mf genre leakage: {sorted(g_ov)}", r, "genre_disjoint_train_test")
        return _common_checks(df, test, side, sup, r, hashes, max_hamming, artist_rules=False)
    if protocol in NEEDS_ARTIST_DISJOINT:
        _check(not a_ov, f"artist leakage train/test: {sorted(a_ov)[:5]}", r, "artist_disjoint_train_test")
        _check(not (set(val["artist"]) & set(fit["artist"])), "val artists overlap fit artists", r,
               "artist_disjoint_val_fit")
    if protocol in NEEDS_GENRE_DISJOINT:
        _check(not g_ov, f"genre leakage train/test: {sorted(g_ov)}", r, "genre_disjoint_train_test")
        _check(not (set(val["genre"]) & set(fit["genre"])), "val genres overlap fit genres", r,
               "genre_disjoint_val_fit")
        _check(test["genre"].notna().all() and side["genre"].notna().all(), "missing genre in genre split",
               r, "genre_known")
    return _common_checks(df, test, side, sup, r, hashes, max_hamming, protocol in NEEDS_ARTIST_DISJOINT)


def _common_checks(df, test, side, sup, r, hashes, max_hamming, artist_rules):
    _check(set(test["style"]) == set(side["style"]), "style sets differ between train side and test", r,
           "styles_shared")
    tr_n = side.groupby("style").size()
    te_n = test.groupby("style").size()
    _check((tr_n >= sup["min_train_per_style"]).all() and (te_n >= sup["min_test_per_style"]).all(),
           "per-style support violated", r, "support_counts")
    if artist_rules:
        _check((side.groupby("style")["artist"].nunique() >= sup["min_train_artists_per_style"]).all() and
               (test.groupby("style")["artist"].nunique() >= sup["min_test_artists_per_style"]).all(),
               "per-style artist support violated", r, "support_artists")
    if hashes is not None:
        h = hashes.set_index("image_id").loc[df.image_id]
        _check(h.sha256.is_unique, "exact duplicate inside split", r, "no_exact_duplicates")
        ph = np.array([int(x, 16) for x in h.phash], dtype=np.uint64)
        is_test = (df.part == "test").to_numpy()
        cross = [(i, j) for i, j, _ in near_duplicate_pairs(ph, max_hamming) if is_test[i] != is_test[j]]
        r["near_duplicate_cross_pairs"] = len(cross)
        _check(not cross, f"{len(cross)} near-duplicate pairs cross train/test", r, "no_cross_near_duplicates")
    return r
