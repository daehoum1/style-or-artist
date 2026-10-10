"""Mechanistic diagnostic on the IID protocol (kNN-5): how often the query's neighbours share its artist /
genre, and accuracy conditional on that. Output: artifacts/results/neighbor_diag/<ds>.csv"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS  # noqa: E402
from stylebench.features.cache import available, load_embeddings  # noqa: E402
from stylebench.models.base import encoder_configs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
a = ap.parse_args()
rows = []
dev = "cuda" if torch.cuda.is_available() else "cpu"
for e in encoder_configs():
    if not available(a.dataset, e):
        continue
    E, row, _ = load_embeddings(a.dataset, e)
    for f in sorted((ARTIFACTS / "splits" / a.dataset / "iid").glob("seed*.parquet")):
        sp = pd.read_parquet(f)
        tr, te = sp[sp.part != "test"], sp[sp.part == "test"]
        A = torch.as_tensor(E[tr.image_id.map(row).to_numpy()], device=dev)
        B = torch.as_tensor(E[te.image_id.map(row).to_numpy()], device=dev)
        idx = torch.cat([(B[i:i + 4096] @ A.T).topk(5, dim=1).indices.cpu() for i in range(0, len(B), 4096)]).numpy()
        nst = tr["style"].to_numpy()[idx]
        nar = tr["artist"].to_numpy()[idx]
        qst, qar = te["style"].to_numpy(), te["artist"].to_numpy()
        # majority vote; ties broken by the nearest tied neighbour (close to kNN-5's similarity tie-break)
        cls, inv = np.unique(np.r_[tr["style"].to_numpy(), qst], return_inverse=True)
        ni = inv[: len(tr)][idx]
        votes = np.zeros((len(te), len(cls)))
        np.add.at(votes, (np.arange(len(te))[:, None].repeat(5, 1), ni), 1.0 + 1e-3 * np.linspace(1, 0.5, 5)[None, :])
        correct = votes.argmax(1) == inv[len(tr):]
        same_art_top1 = nar[:, 0] == qar
        same_art_any = (nar == qar[:, None]).any(1)
        r = {"encoder": e, "seed": int(f.stem[4:]), "acc": correct.mean(),
             "share_top1_same_artist": same_art_top1.mean(), "share_any5_same_artist": same_art_any.mean(),
             "mean_same_artist_in_5": (nar == qar[:, None]).sum(1).mean(),
             "acc_if_top1_same_artist": correct[same_art_top1].mean(),
             "acc_if_top1_other_artist": correct[~same_art_top1].mean(),
             "acc_if_no_same_artist_in_5": correct[~same_art_any].mean()}
        if te["genre"].notna().all():
            ngen = tr["genre"].to_numpy()[idx]
            qg = te["genre"].to_numpy()
            sg = ngen[:, 0] == qg
            r.update({"share_top1_same_genre": sg.mean(), "acc_if_top1_same_genre": correct[sg].mean(),
                      "acc_if_top1_other_genre": correct[~sg].mean(),
                      "share_top1_same_genre_other_artist": (sg & ~same_art_top1).mean()})
        rows.append(r)
out = ARTIFACTS / "results" / "neighbor_diag"
out.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / f"{a.dataset}.csv", index=False)
print(df.groupby("encoder").mean(numeric_only=True).drop(columns="seed").round(3).to_string())
