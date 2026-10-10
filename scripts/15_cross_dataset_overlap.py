"""Revision: overlap between WikiArt and MultitaskPainting100k (cleaned sets and controlled pools):
exact (SHA-256) and perceptual (pHash Hamming <= 4) artwork overlap, artist-name overlap, style-name overlap.
Output: artifacts/reports/cross_dataset_overlap.json"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, split_config  # noqa: E402
from stylebench.data.hashing import near_duplicate_pairs  # noqa: E402
from stylebench.data.splits import controlled_pool  # noqa: E402


def key(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


M = ARTIFACTS / "metadata"
cw, cm = pd.read_parquet(M / "wikiart" / "cleaned.parquet"), pd.read_parquet(M / "multitask_painting" / "cleaned.parquet")
sup = split_config()["support"]
pw, _ = controlled_pool(cw, sup)
pm, _ = controlled_pool(cm, sup)
out = {}
for name, a, b in [("cleaned", cw, cm), ("controlled_pool", cw[cw.image_id.isin(pw.image_id)], cm[cm.image_id.isin(pm.image_id)])]:
    sha = set(a.sha256) & set(b.sha256)
    ph = np.r_[np.array([int(h, 16) for h in a.phash], dtype=np.uint64), np.array([int(h, 16) for h in b.phash], dtype=np.uint64)]
    pairs = near_duplicate_pairs(ph, 4)
    na = len(a)
    cross = [(i, j - na) for i, j, _ in pairs if i < na <= j]
    wa_hit = len({i for i, _ in cross} | set(np.flatnonzero(a.sha256.isin(sha))))
    mp_hit = len({j for _, j in cross} | set(np.flatnonzero(b.sha256.isin(sha))))
    aw, am_ = {key(x) for x in a["artist"]}, {key(x) for x in b["artist"]}
    sw, sm = {key(x) for x in a["style"]}, {key(x) for x in b["style"]}
    out[name] = {"n_wikiart": int(na), "n_mp": int(len(b)), "exact_shared_images": len(sha),
                 "wikiart_images_with_mp_duplicate": wa_hit, "share_wikiart": round(wa_hit / na, 3),
                 "mp_images_with_wikiart_duplicate": mp_hit, "share_mp": round(mp_hit / len(b), 3),
                 "artists_wikiart": len(aw), "artists_mp": len(am_), "artists_shared": len(aw & am_),
                 "share_wikiart_artists_in_mp": round(len(aw & am_) / len(aw), 3),
                 "styles_wikiart": len(sw), "styles_mp": len(sm), "styles_shared_by_name": sorted(sw & sm)}
json.dump(out, open(ARTIFACTS / "reports" / "cross_dataset_overlap.json", "w"), indent=1)
print(json.dumps(out, indent=1))
