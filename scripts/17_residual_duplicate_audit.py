"""Revision: residual near-duplicate audit of the main splits. For every protocol and seed, test images are
flagged if a training image is within pHash Hamming distance 6, 8 or 10 (the cleaning used 4), or if their
DINOv2-B/14 cosine similarity to some training image is >= 0.95. kNN-5 balanced accuracy of every encoder is
then recomputed without the flagged test images (flag = pHash <= 8 or cosine >= 0.95).
Output: artifacts/results/dup_audit/<ds>_flags.csv, <ds>_effect.csv"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS  # noqa: E402
from stylebench.data.hashing import near_duplicate_pairs  # noqa: E402
from stylebench.evaluation.classifiers import knn_predict  # noqa: E402
from stylebench.evaluation.metrics import confusion, metrics_from_confusion  # noqa: E402
from stylebench.features.cache import available, load_embeddings  # noqa: E402
from stylebench.models.base import encoder_configs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
a = ap.parse_args()
H = pd.read_parquet(ARTIFACTS / "metadata" / a.dataset / "hashes.parquet").set_index("image_id")
Ed, rowd, _ = load_embeddings(a.dataset, "dinov2_b14")
encs = [e for e in encoder_configs() if available(a.dataset, e)]
EMB = {}
for e in encs:  # float16 copies to bound memory
    _E, _r, _ = load_embeddings(a.dataset, e)
    EMB[e] = (_E.astype(np.float16), _r)
flags, eff = [], []
for p in ["iid", "artist_disjoint", "genre_disjoint", "joint"]:
    for f in sorted((ARTIFACTS / "splits" / a.dataset / p).glob("seed*.parquet")):
        seed = int(f.stem[4:])
        sp = pd.read_parquet(f)
        is_te = (sp.part == "test").to_numpy()
        ph = np.array([int(x, 16) for x in H.loc[sp.image_id, "phash"]], dtype=np.uint64).view(np.int64)
        # exact cross-split minimum Hamming distance on the GPU (SWAR popcount), test x train
        P_tr = torch.as_tensor(ph[~is_te], device="cuda")
        P_te = torch.as_tensor(ph[is_te], device="cuda")
        mind = []
        for i in range(0, len(P_te), 1024):
            x = P_te[i:i + 1024, None] ^ P_tr[None, :]
            x = x - ((x >> 1) & 0x5555555555555555)
            x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
            x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
            cnt = ((x * 0x0101010101010101) >> 56) & 0xFF
            mind.append(cnt.min(1).values.cpu())
        mind = torch.cat(mind).numpy()
        fl = {}
        for t in (6, 8, 10):
            m = np.zeros(len(sp), bool)
            m[np.flatnonzero(is_te)] = mind <= t
            fl[f"phash{t}"] = m
        A = torch.as_tensor(Ed[sp.image_id[~is_te].map(rowd).to_numpy()], device="cuda")
        B = torch.as_tensor(Ed[sp.image_id[is_te].map(rowd).to_numpy()], device="cuda")
        mx = torch.cat([(B[i:i + 4096] @ A.T).max(1).values.cpu() for i in range(0, len(B), 4096)]).numpy()
        cos = np.zeros(len(sp), bool)
        cos[np.flatnonzero(is_te)] = mx >= 0.95
        fl["dino_cos95"] = cos
        flag = fl["phash8"] | fl["dino_cos95"]
        flags.append({"protocol": p, "seed": seed, "n_test": int(is_te.sum()),
                      **{k: int(v.sum()) for k, v in fl.items()}, "flag_union": int(flag.sum())})
        styles = sorted(sp["style"].unique())
        y = sp["style"].map({s: i for i, s in enumerate(styles)}).to_numpy()
        keep_te = is_te & ~flag
        for e in encs:
            E, row = EMB[e]
            X = E[sp.image_id.map(row).to_numpy()].astype(np.float32)
            pr = knn_predict(X[~is_te], y[~is_te], X[is_te], [5], len(styles))[5]
            sub = ~flag[is_te]
            b0 = metrics_from_confusion(confusion(y[is_te], pr, len(styles)))["balanced_accuracy"]
            b1 = metrics_from_confusion(confusion(y[is_te][sub], pr[sub], len(styles)))["balanced_accuracy"]
            eff.append({"protocol": p, "seed": seed, "encoder": e, "bacc_all": b0, "bacc_unflagged": b1})
out = ARTIFACTS / "results" / "dup_audit"
out.mkdir(parents=True, exist_ok=True)
F, X = pd.DataFrame(flags), pd.DataFrame(eff)
F.to_csv(out / f"{a.dataset}_flags.csv", index=False)
X.to_csv(out / f"{a.dataset}_effect.csv", index=False)
print(F.groupby("protocol").mean(numeric_only=True).round(1).to_string())
X["delta"] = 100 * (X.bacc_unflagged - X.bacc_all)
print(X.groupby("protocol").delta.agg(["mean", "min", "max"]).round(2).to_string())
