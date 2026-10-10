"""Revision round 3 (R2-C4): exposure-matched effects per test artist (kNN-5, per-artist accuracy) by the
artist's number of test paintings (terciles within dataset and encoder), to explain why artist-macro effects
are smaller than style-macro effects. Output: artifacts/final/rev3/artist_size_effects.csv"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS  # noqa: E402

rows = []
for ds in ["wikiart", "multitask_painting"]:
    for f in sorted((ARTIFACTS / "results" / "preds" / ds).glob("*.parquet")):
        P = pd.read_parquet(f)
        P = P[(P.evaluator == "knn5") & P.protocol.str.startswith("emf_")].assign(ok=lambda d: d.y == d.pred)
        a = P.pivot_table(index=["seed", "artist"], columns="protocol", values="ok", aggfunc="mean")
        a["n"] = P[P.protocol == "emf_AG"].groupby(["seed", "artist"]).size()
        a["interaction"] = (a.emf_AG - a.emf_A) - (a.emf_G - a.emf_none)
        a["genre_Aminus"] = a.emf_G - a.emf_none
        a["tercile"] = pd.qcut(a.n.rank(method="first"), 3, labels=["small", "mid", "large"])
        g = a.groupby("tercile", observed=True)
        for t, r in g[["interaction", "genre_Aminus", "n"]].mean().iterrows():
            rows.append({"dataset": ds, "encoder": f.stem, "tercile": t, **r.to_dict()})
R = pd.DataFrame(rows)
R.to_csv(ARTIFACTS / "final" / "rev3" / "artist_size_effects.csv", index=False)
print(R.groupby(["dataset", "tercile"])[["interaction", "genre_Aminus", "n"]].mean().round(3))
