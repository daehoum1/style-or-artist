"""Revision round 3 (R2-C2): representativeness of the artists evaluated in the exposure-matched design.
A test artist of the Joint partition enters the exposure-matched test set only if, in some style, it has >= 2
images in the held-out genres (TT) and >= 1 in the training genres (TA). This script compares, per seed, the
eligible (evaluated) and ineligible test artists of the Joint partition on
  * pool counts: images, styles, genres, share of images in the held-out genres,
  * artist-macro kNN-5 accuracy on the IID, Artist-D and Joint test sets of the same seed (an artist enters a
    protocol's comparison only where it is on that protocol's test side),
and re-estimates the exposure-matched effects (kNN-5, style-balanced accuracy) separately for evaluated artists
with few and with many genres (median split within each seed).
Output: artifacts/final/rev3/emf_eligibility.csv, emf_eligibility_summary.csv, emf_breadth_effects.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, split_config  # noqa: E402
from stylebench.data.splits import controlled_pool  # noqa: E402
from stylebench.evaluation.classifiers import knn_predict  # noqa: E402
from stylebench.features.cache import available, load_embeddings  # noqa: E402

ENC = ["clip_b16", "clip_l14", "openclip_b16", "siglip_b16", "siglip2_l16", "siglip_so400m", "dinov2_b14",
       "dinov2_l14", "dinov3_b16", "mae_b16", "vit_b16_in21k", "csd_l14", "aladin_vit"]
OUT = ARTIFACTS / "final" / "rev3"
OUT.mkdir(parents=True, exist_ok=True)
cfg = split_config()
rows, eff_rows = [], []
for ds in ["wikiart", "multitask_painting"]:
    root = ARTIFACTS / "splits" / ds
    clean = pd.read_parquet(ARTIFACTS / "metadata" / ds / "cleaned.parquet")
    pool, _ = controlled_pool(clean, cfg["support"])
    splits = {(p, s): pd.read_parquet(root / p / f"seed{s}.parquet")
              for p in ["iid", "artist_disjoint", "joint", "emf_AG"] for s in cfg["seeds"]}
    for enc in ENC:
        if not available(ds, enc):
            continue
        E, row, _ = load_embeddings(ds, enc)
        P = pd.read_parquet(ARTIFACTS / "results" / "preds" / ds / f"{enc}.parquet")
        P = P[(P.evaluator == "knn5") & P.protocol.str.startswith("emf_")]
        for s in cfg["seeds"]:
            jt = splits[("joint", s)]
            G_te = set(jt[jt.part == "test"]["genre"])
            pop = sorted(set(jt[jt.part == "test"]["artist"]))
            emf_te = splits[("emf_AG", s)]
            elig = set(emf_te[emf_te.part == "test"]["artist"])
            acc = {}
            for p in ["iid", "artist_disjoint", "joint"]:
                sp = splits[(p, s)]
                styles = sorted(sp["style"].unique())
                y = sp["style"].map({v: i for i, v in enumerate(styles)}).to_numpy()
                X = E[sp.image_id.map(row).to_numpy()]
                tr, te = sp.part.isin(["train", "val"]).to_numpy(), (sp.part == "test").to_numpy()
                pr = knn_predict(X[tr], y[tr], X[te], [5], len(styles))[5]
                acc[p] = pd.Series(pr == y[te]).groupby(sp["artist"][te].values).mean()
            pp = pool[pool["artist"].isin(pop)]
            g = pp.groupby("artist")
            art = pd.DataFrame({"n_images": g.size(), "n_styles": g["style"].nunique(), "n_genres": g["genre"].nunique(),
                                "heldout_genre_share": g["genre"].apply(lambda x: x.isin(G_te).mean())})
            for p, v in acc.items():
                art[f"acc_{p}"] = v.reindex(art.index)
            art["eligible"] = art.index.isin(elig)
            art = art.assign(dataset=ds, encoder=enc, seed=s).reset_index().rename(columns={"index": "artist"})
            rows.append(art)
            # exposure-matched effects for evaluated artists with few vs many genres (median split within seed)
            ea = art[art.eligible].set_index("artist")["n_genres"]
            med = ea.median()
            for grp, members in [("few_genres", set(ea[ea <= med].index)), ("many_genres", set(ea[ea > med].index))]:
                q = P[(P.seed == s) & P.artist.isin(members)]
                Y = {c: q[q.protocol == c].assign(ok=lambda d: d.y == d.pred).groupby("style").ok.mean().mean()
                     for c in ["emf_AG", "emf_A", "emf_G", "emf_none"]}
                eff_rows.append({"dataset": ds, "encoder": enc, "seed": s, "group": grp, "n_artists": len(members),
                                 "median_genres": med, "genre_Aplus": Y["emf_AG"] - Y["emf_A"],
                                 "genre_Aminus": Y["emf_G"] - Y["emf_none"], "artist": Y["emf_AG"] - Y["emf_G"],
                                 "interaction": (Y["emf_AG"] - Y["emf_A"]) - (Y["emf_G"] - Y["emf_none"])})
        print(ds, enc, flush=True)
A = pd.concat(rows, ignore_index=True)
A.to_csv(OUT / "emf_eligibility.csv", index=False)
cols = ["n_images", "n_styles", "n_genres", "heldout_genre_share", "acc_iid", "acc_artist_disjoint", "acc_joint"]
# per (dataset, encoder, seed, eligible): artist-macro means; then mean over seeds and encoders
m = A.groupby(["dataset", "encoder", "seed", "eligible"])[cols].mean()
m["n_artists"] = A.groupby(["dataset", "encoder", "seed", "eligible"]).size()
summ = m.groupby(["dataset", "eligible"]).mean()
d = m.xs(True, level="eligible")[cols] - m.xs(False, level="eligible")[cols]
dd = d.groupby("dataset").agg(["mean", "min", "max"])
summ.to_csv(OUT / "emf_eligibility_summary.csv")
dd.to_csv(OUT / "emf_eligibility_diff.csv")
B = pd.DataFrame(eff_rows)
B.to_csv(OUT / "emf_breadth_effects.csv", index=False)
print(summ.round(3).T)
print(dd.round(3).T)
print(B.groupby(["dataset", "group"])[["n_artists", "median_genres", "artist", "genre_Aplus", "genre_Aminus",
                                        "interaction"]].mean().round(3))
