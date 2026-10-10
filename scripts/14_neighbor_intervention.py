"""Neighbour intervention on IID test sets (revision rounds 1-2). For every test painting, k-NN is recomputed
after removing from the candidate training paintings:
  same_artist        : all paintings by the same artist (in any style)
  same_artist_style  : paintings by the same artist in the same style only
  control_random     : as many randomly chosen same-style paintings by other artists (count-matched)
  control_nearest    : as many of the NEAREST same-style paintings by other artists (count- and rank-matched)
for k in {1, 3, 5, 10, 20}; same test images in every condition.
Output: artifacts/results/neighbor_intervention/<ds>.csv (k=5 rows keep the round-1 columns)"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS  # noqa: E402
from stylebench.evaluation.metrics import confusion, metrics_from_confusion  # noqa: E402
from stylebench.features.cache import available, load_embeddings  # noqa: E402
from stylebench.models.base import encoder_configs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
a = ap.parse_args()
dev = "cuda"
KS = [1, 3, 5, 10, 20]
rows = []
for e in encoder_configs():
    if not available(a.dataset, e):
        continue
    E, row, _ = load_embeddings(a.dataset, e)
    for f in sorted((ARTIFACTS / "splits" / a.dataset / "iid").glob("seed*.parquet")):
        seed = int(f.stem[4:])
        sp = pd.read_parquet(f)
        tr, te = sp[sp.part != "test"], sp[sp.part == "test"]
        styles = sorted(sp["style"].unique())
        sid = {s: i for i, s in enumerate(styles)}
        K = len(styles)
        A = torch.as_tensor(E[tr.image_id.map(row).to_numpy()], device=dev)
        B = torch.as_tensor(E[te.image_id.map(row).to_numpy()], device=dev)
        arts = pd.factorize(pd.concat([tr["artist"], te["artist"]]))[0]
        a_tr, a_te = torch.as_tensor(arts[: len(tr)], device=dev), torch.as_tensor(arts[len(tr):], device=dev)
        y_tr = torch.as_tensor(tr["style"].map(sid).to_numpy(), device=dev)
        y_te = te["style"].map(sid).to_numpy()
        g = torch.Generator(device=dev).manual_seed(seed)
        conds = ["none", "same_artist", "same_artist_style", "control_random", "control_nearest"]
        preds = {(c, k): [] for c in conds for k in KS}
        nrem = []
        for i in range(0, len(B), 1024):
            S = B[i:i + 1024] @ A.T
            yq = torch.as_tensor(y_te[i:i + 1024], device=dev)
            same = a_te[i:i + 1024, None] == a_tr[None, :]
            same_sty = same & (y_tr[None, :] == yq[:, None])
            m = same.sum(1)
            elig = (y_tr[None, :] == yq[:, None]) & ~same
            key = torch.rand(S.shape, generator=g, device=dev)
            key[~elig] = 2.0
            thr = key.sort(1).values.gather(1, (m.clamp(min=1) - 1)[:, None])
            ctrl_r = (key <= thr) & (m[:, None] > 0) & elig
            se = S.masked_fill(~elig, -9.0)
            thr2 = se.sort(1, descending=True).values.gather(1, (m.clamp(min=1) - 1)[:, None])
            ctrl_n = (se >= thr2) & (m[:, None] > 0) & elig
            masks = {"none": None, "same_artist": same, "same_artist_style": same_sty,
                     "control_random": ctrl_r, "control_nearest": ctrl_n}
            for c, mask in masks.items():
                SS = S if mask is None else S.masked_fill(mask, -9.0)
                sim, idx = SS.topk(max(KS), 1)
                lab = y_tr[idx]
                for k in KS:
                    votes = torch.zeros(len(lab), K, device=dev).scatter_add_(1, lab[:, :k], torch.ones_like(sim[:, :k]))
                    ssum = torch.zeros(len(lab), K, device=dev).scatter_add_(1, lab[:, :k], sim[:, :k])
                    preds[(c, k)].append((votes + 1e-3 * ssum).argmax(1).cpu().numpy())
            nrem.append(m.cpu().numpy())
        nrem = np.concatenate(nrem)
        has = nrem > 0
        for (c, k), pr in preds.items():
            pr = np.concatenate(pr)
            m_all = metrics_from_confusion(confusion(y_te, pr, K))
            rows.append({"encoder": e, "seed": seed, "k": k, "condition": c, "bacc": m_all["balanced_accuracy"],
                         "acc": m_all["accuracy"], "share_with_same_artist": has.mean(), "mean_removed": nrem[has].mean()})
out = ARTIFACTS / "results" / "neighbor_intervention"
out.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / f"{a.dataset}.csv", index=False)
print((100 * df[df.k == 5].groupby(["encoder", "condition"]).bacc.mean().unstack()).round(1).to_string())
print((100 * df.groupby(["k", "condition"]).bacc.mean().unstack()).round(1).to_string())
