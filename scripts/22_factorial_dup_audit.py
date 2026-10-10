"""Revision round 2 (C6): near-duplicates between the test set and the training images of the 'test artists and
test genres seen' condition (exposure-matched add-on images; matched-design training set), which come from the
same (artist, style, test-genre) cells as the test images. Flags test images with a training image at pHash
Hamming distance <= 8 or DINOv2-B/14 cosine >= 0.95.
Output: artifacts/results/rev2/factorial_dup_flags.csv (dataset, seed, design, image_id) + summary csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS  # noqa: E402
from stylebench.features.cache import load_embeddings  # noqa: E402


def min_hamming(te, tr):
    T, P = torch.as_tensor(te, device="cuda"), torch.as_tensor(tr, device="cuda")
    out = []
    for i in range(0, len(T), 1024):
        x = T[i:i + 1024, None] ^ P[None, :]
        x = x - ((x >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
        out.append((((x * 0x0101010101010101) >> 56) & 0xFF).min(1).values.cpu())
    return torch.cat(out).numpy()


flags, summ = [], []
for ds, base in [("wikiart", "wikiart"), ("multitask_painting", "multitask_painting"),
                 ("wikiart_ga", "wikiart"), ("multitask_painting_ga", "multitask_painting")]:
    H = pd.read_parquet(ARTIFACTS / "metadata" / base / "hashes.parquet").set_index("image_id")
    E, row, _ = load_embeddings(base, "dinov2_b14")
    for design, proto, sel in [("emf", "emf_AG", lambda d: d.role == "addon"), ("mf", "mf_iid", lambda d: d.part != "test")]:
        for f in sorted((ARTIFACTS / "splits" / ds / proto).glob("seed*.parquet")):
            seed = int(f.stem[4:])
            sp = pd.read_parquet(f)
            te, tr = sp[sp.part == "test"], sp[sel(sp)]
            ph = lambda ids: np.array([int(x, 16) for x in H.loc[ids, "phash"]], dtype=np.uint64).view(np.int64)  # noqa: E731
            mh = min_hamming(ph(te.image_id), ph(tr.image_id))
            A = torch.as_tensor(E[tr.image_id.map(row).to_numpy()], device="cuda")
            B = torch.as_tensor(E[te.image_id.map(row).to_numpy()], device="cuda")
            mc = torch.cat([(B[i:i + 4096] @ A.T).max(1).values.cpu() for i in range(0, len(B), 4096)]).numpy()
            fl = (mh <= 8) | (mc >= 0.95)
            for i in te.image_id[fl]:
                flags.append({"dataset": ds, "seed": seed, "design": design, "image_id": i})
            summ.append({"dataset": ds, "design": design, "seed": seed, "n_test": len(te), "phash8": int((mh <= 8).sum()),
                         "cos95": int((mc >= 0.95).sum()), "flagged": int(fl.sum())})
out = ARTIFACTS / "results" / "rev2"
pd.DataFrame(flags).to_csv(out / "factorial_dup_flags.csv", index=False)
S = pd.DataFrame(summ)
S.to_csv(out / "factorial_dup_summary.csv", index=False)
g = S.groupby(["dataset", "design"])[["n_test", "phash8", "cos95", "flagged"]].mean()
g["pct"] = 100 * g.flagged / g.n_test
print(g.round(2).to_string())
