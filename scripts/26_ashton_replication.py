"""Revision round 2 (C10): the setting of Ashton (2026) reproduced on MultitaskPainting100k: four movements
(Abstract Expressionism, Cubism incl. Analytical/Synthetic Cubism, Impressionism, Surrealism), 8 artists per
movement with >= 10 works, 10 works per artist (320 images); 5-NN accuracy with the full pool (leave-one-image-out;
same-artist works allowed as neighbours) vs leave-one-artist-out; 50 random draws; all encoders.
Output: artifacts/results/rev2/ashton_replication.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS  # noqa: E402
from stylebench.features.cache import load_embeddings  # noqa: E402
from stylebench.models.base import encoder_configs  # noqa: E402

MOV = {"abstract expressionism": "Abstract Expressionism", "cubism": "Cubism", "analytical cubism": "Cubism",
       "synthetic cubism": "Cubism", "impressionism": "Impressionism", "surrealism": "Surrealism"}
c = pd.read_parquet(ARTIFACTS / "metadata" / "multitask_painting" / "cleaned.parquet")
c = c[c["style"].isin(MOV)].assign(mov=lambda x: x["style"].map(MOV))
draws = []
rng = np.random.default_rng(2026)
for r in range(50):
    parts = []
    for m, g in c.groupby("mov"):
        vc = g.artist.value_counts()
        arts = rng.choice(vc.index[vc >= 10], 8, replace=False)
        for a in arts:
            ga = g[g.artist == a]
            parts.append(ga.iloc[rng.choice(len(ga), 10, replace=False)])
    draws.append(pd.concat(parts).reset_index(drop=True))
rows = []
for e in encoder_configs():
    E, row, _ = load_embeddings("multitask_painting", e)
    for r, d in enumerate(draws):
        X = E[d.image_id.map(row).to_numpy()]
        S = X @ X.T
        np.fill_diagonal(S, -9)
        y = d.mov.to_numpy()
        art = d.artist.to_numpy()
        for proto in ["full_pool", "leave_one_artist_out"]:
            SS = S.copy()
            if proto == "leave_one_artist_out":
                SS[art[:, None] == art[None, :]] = -9
            idx = np.argsort(-SS, axis=1)[:, :5]
            pred = []
            for i in range(len(d)):
                lab, cnt = np.unique(y[idx[i]], return_counts=True)
                best = lab[cnt == cnt.max()]
                pred.append(best[0] if len(best) == 1 else next(y[j] for j in idx[i] if y[j] in best))
            pred = np.array(pred)
            for m in sorted(set(y)):
                rows.append({"encoder": e, "draw": r, "protocol": proto, "movement": m, "acc": (pred[y == m] == m).mean()})
            rows.append({"encoder": e, "draw": r, "protocol": proto, "movement": "all", "acc": (pred == y).mean()})
df = pd.DataFrame(rows)
df.to_csv(ARTIFACTS / "results" / "rev2" / "ashton_replication.csv", index=False)
print((100 * df.groupby(["movement", "protocol"]).acc.mean().unstack()).round(1).to_string())
print((100 * df[df.movement == "all"].groupby(["encoder", "protocol"]).acc.mean().unstack()).round(1).to_string())
