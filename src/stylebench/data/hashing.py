"""Exact (SHA-256 of file bytes) and perceptual (64-bit DCT pHash) hashes, computed in parallel and cached."""
from __future__ import annotations

import hashlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

Image.MAX_IMAGE_PIXELS = None


def phash64(img: Image.Image) -> int:
    """Standard pHash: 32x32 grayscale -> 2D DCT -> top-left 8x8 (incl. DC) -> threshold at median."""
    g = img.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    d = dctn(np.asarray(g, dtype=np.float64), norm="ortho")[:8, :8].ravel()
    bits = d > np.median(d)
    return int(np.packbits(bits).view(">u8")[0])


def _hash_one(path: str):
    try:
        with open(path, "rb") as f:
            b = f.read()
        sha = hashlib.sha256(b).hexdigest()
        with Image.open(path) as im:
            w, h = im.size
            im.draft("RGB", (256, 256))  # fast JPEG DCT-domain downscale; deterministic
            ph = phash64(im)
        return sha, f"{ph:016x}", w, h, True, ""
    except Exception as e:  # unreadable images are recorded, not silently dropped
        return None, None, None, None, False, repr(e)[:200]


def compute_hashes(paths: list[str], workers: int = 20) -> pd.DataFrame:
    with Pool(workers) as p:
        out = p.map(_hash_one, paths, chunksize=64)
    return pd.DataFrame(out, columns=["sha256", "phash", "width", "height", "readable", "read_error"])


def near_duplicate_pairs(phashes: np.ndarray, max_dist: int) -> list[tuple[int, int, int]]:
    """All pairs with Hamming distance <= max_dist (pigeonhole on max_dist+1 bit chunks; exact)."""
    n_chunks = max_dist + 1
    edges = np.linspace(0, 64, n_chunks + 1).astype(int)
    vals = phashes.astype(np.uint64)
    cand = set()
    for a, b in zip(edges[:-1], edges[1:]):
        key = (vals >> np.uint64(64 - b)) & np.uint64((1 << (b - a)) - 1)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[starts[1:], len(ks)]
        for s, e in zip(starts, ends):
            if e - s > 1:
                idx = order[s:e]
                if e - s > 2000:  # degenerate bucket (e.g., blank images); still exact, just slower
                    pass
                for i in range(len(idx)):
                    for j in range(i + 1, len(idx)):
                        x, y = idx[i], idx[j]
                        cand.add((min(x, y), max(x, y)))
    out = []
    for i, j in cand:
        d = bin(int(vals[i] ^ vals[j])).count("1")
        if d <= max_dist:
            out.append((i, j, d))
    return out
