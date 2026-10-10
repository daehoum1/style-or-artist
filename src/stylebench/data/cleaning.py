"""Cleaning: unreadable files, exact/near duplicate grouping, label conflicts. Every decision is logged."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .hashing import near_duplicate_pairs


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _uniq(s: pd.Series) -> list:
    return sorted(set(s.dropna().tolist()))


def clean(raw: pd.DataFrame, hashes: pd.DataFrame, max_hamming: int, conflict_style_policy: str = "drop"):
    """conflict_style_policy: 'drop' (main) removes duplicate groups with conflicting styles; 'keep_representative'
    (sensitivity variant) keeps one copy (largest image) with its own label."""
    df = raw.merge(hashes, on="image_id", how="left", validate="one_to_one")
    decisions = []
    bad = ~df.readable.fillna(False).astype(bool)
    for iid, err in zip(df.image_id[bad], df.read_error[bad]):
        decisions.append((iid, "drop", "unreadable", str(err)))
    df = df[~bad].reset_index(drop=True)

    n = len(df)
    uf = UnionFind(n)
    # exact duplicates
    for _, idx in df.groupby("sha256").indices.items():
        for j in idx[1:]:
            uf.union(idx[0], j)
    # near duplicates
    ph = np.array([int(h, 16) for h in df.phash], dtype=np.uint64)
    pairs = near_duplicate_pairs(ph, max_hamming)
    exact_pairs = 0
    for i, j, d in pairs:
        uf.union(i, j)
        exact_pairs += d == 0
    df["dup_group"] = [uf.find(i) for i in range(n)]
    df["dup_group_size"] = df.groupby("dup_group").image_id.transform("size")
    sha_size = df.groupby("sha256").image_id.transform("size")
    df["has_exact_dup"] = sha_size > 1

    keep = np.ones(n, bool)
    genre_conflict = np.zeros(n, bool)
    for g, sub in df[df.dup_group_size > 1].groupby("dup_group"):
        idx = sub.index.to_numpy()  # df has a RangeIndex, so labels == positions
        ids = sub.image_id.tolist()
        artists, styles, genres = _uniq(sub["artist"]), _uniq(sub["style"]), _uniq(sub["genre"])
        kind = "exact" if sub.sha256.nunique() == 1 else "near"
        if len(artists) > 1:
            keep[idx] = False
            for i in ids:
                decisions.append((i, "drop", f"{kind}_duplicate_group_conflicting_artist", "|".join(artists)))
            continue
        if len(styles) > 1 and conflict_style_policy == "drop":
            keep[idx] = False
            for i in ids:
                decisions.append((i, "drop", f"{kind}_duplicate_group_conflicting_style", "|".join(styles)))
            continue
        order = sub.assign(px=sub.width * sub.height).sort_values(["px", "image_id"], ascending=[False, True])
        rep = order.index[0]
        for r in order.index[1:]:
            keep[r] = False
            decisions.append((df.image_id[r], "drop", f"{kind}_duplicate_of", df.image_id[rep]))
        # labels missing on the representative but present (and unique) in the group are NOT copied:
        # keep the representative's own labels only, to avoid creating metadata.
        if len(genres) > 1:
            genre_conflict[rep] = True
            decisions.append((df.image_id[rep], "modify", f"{kind}_duplicate_group_conflicting_genre",
                              "|".join(genres)))
    df.loc[genre_conflict, "genre"] = pd.NA
    cleaned = df[keep].reset_index(drop=True)
    assert cleaned.sha256.is_unique
    assert cleaned.dup_group.is_unique
    dec = pd.DataFrame(decisions, columns=["image_id", "action", "reason", "detail"])
    stats = {
        "n_raw": int(len(raw)), "n_unreadable": int(bad.sum()), "n_after_clean": int(len(cleaned)),
        "n_exact_duplicate_images": int((sha_size > 1).sum()),
        "n_near_duplicate_pairs": int(len(pairs)), "n_near_pairs_identical_phash": int(exact_pairs),
        "n_dup_groups_multi": int((df.groupby("dup_group").size() > 1).sum()),
        "max_dup_group_size": int(df.dup_group_size.max()),
        "near_duplicate_max_hamming": max_hamming,
    }
    stats["version"] = table_version(cleaned)
    return cleaned, dec, stats


def table_version(df: pd.DataFrame) -> str:
    cols = ["image_id", "style", "artist", "genre", "sha256"]
    h = hashlib.sha256(pd.util.hash_pandas_object(df[cols].astype(str), index=False).values.tobytes())
    return h.hexdigest()[:12]
