"""Revision round 2 (C2): genre-dependent artist selection in the Joint partitions.
For each dataset and seed: share of held-out-genre paintings among test artists' paintings, TT/(TT+TA), and
among training artists' paintings, AT/(AT+BB); the same for the genre-agnostic partition (Artist-D artists x
Genre-D genres of the same seed, used in D1); per-artist distribution of the held-out-genre share.
Output: artifacts/results/rev2/selection_bias.csv, selection_bias_artists.csv"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, split_config  # noqa: E402
from stylebench.data.splits import controlled_pool  # noqa: E402

rows, arts = [], []
for d in ["wikiart", "multitask_painting"]:
    clean = pd.read_parquet(ARTIFACTS / "metadata" / d / "cleaned.parquet")
    pool, _ = controlled_pool(clean, split_config()["support"])
    for seed in range(10):
        jd = json.load(open(ARTIFACTS / "splits" / d / "joint" / f"seed{seed}.json"))
        ad = json.load(open(ARTIFACTS / "splits" / d / "artist_disjoint" / f"seed{seed}.json"))
        styles = set(pd.read_parquet(ARTIFACTS / "splits" / d / "joint" / f"seed{seed}.parquet")["style"])
        p = pool[pool["style"].isin(styles)]
        g = p["genre"].isin(jd["test_genres"])
        base_share = float(g.mean())
        for part, A_te in [("joint", jd["test_artists"]), ("genre_agnostic", ad["test_artists"])]:
            a = p["artist"].isin(A_te)
            TT, TA, AT, BB = (a & g).sum(), (a & ~g).sum(), (~a & g).sum(), (~a & ~g).sum()
            rows.append({"dataset": d, "seed": seed, "partition": part, "pool_heldout_genre_share": base_share,
                         "test_artists_heldout_share": TT / (TT + TA), "train_artists_heldout_share": AT / (AT + BB),
                         "TT": int(TT), "TA": int(TA), "AT": int(AT), "BB": int(BB)})
            sh = p.assign(h=g).groupby("artist").h.mean()
            arts.append(pd.DataFrame({"dataset": d, "seed": seed, "partition": part, "artist": sh.index,
                                      "heldout_share": sh.values, "is_test": sh.index.isin(A_te)}))
out = ARTIFACTS / "results" / "rev2"
R = pd.DataFrame(rows)
R.to_csv(out / "selection_bias.csv", index=False)
pd.concat(arts).to_csv(out / "selection_bias_artists.csv", index=False)
print(R.groupby(["dataset", "partition"])[["pool_heldout_genre_share", "test_artists_heldout_share",
                                           "train_artists_heldout_share"]].mean().round(3).to_string())
