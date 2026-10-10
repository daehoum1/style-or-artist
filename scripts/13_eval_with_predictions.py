"""Revision: evaluate given protocols and SAVE per-image predictions (needed for artist-cluster bootstrap).
Evaluators: kNN-5 (primary), logistic regression (wd on validation, warm-started path), NCC.
Output: artifacts/results/preds/<ds>/<encoder>.parquet (image_id, protocol, seed, evaluator, y, pred)"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, load_yaml  # noqa: E402
from stylebench.evaluation.classifiers import knn_predict, logreg_select_and_fit, ncc_predict  # noqa: E402
from stylebench.evaluation.metrics import confusion, metrics_from_confusion  # noqa: E402
from stylebench.features.cache import available, load_embeddings  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("encoders", nargs="+")
ap.add_argument("--protocols", default="mf_iid,mf_artist,mf_genre,mf_joint,emf_AG,emf_A,emf_G,emf_none")
ap.add_argument("--no_logreg", action="store_true")
ap.add_argument("--only_logreg", action="store_true", help="logistic regression only, written to results/preds_lr/")
a = ap.parse_args()
ecfg = load_yaml("configs/experiments/evaluation.yaml")
lg = ecfg["evaluators"]["logreg"]
out = ARTIFACTS / "results" / ("preds_lr" if a.only_logreg else "preds") / a.dataset
out.mkdir(parents=True, exist_ok=True)
for e in a.encoders:
    if not available(a.dataset, e) or (out / f"{e}.parquet").exists():
        continue
    E, row, _ = load_embeddings(a.dataset, e)
    recs = []
    for p in a.protocols.split(","):
        for f in sorted((ARTIFACTS / "splits" / a.dataset / p).glob("seed*.parquet")):
            seed = int(f.stem[4:])
            sp = pd.read_parquet(f)
            styles = sorted(sp["style"].unique())
            K = len(styles)
            y = sp["style"].map({s: i for i, s in enumerate(styles)}).to_numpy()
            X = E[sp.image_id.map(row).to_numpy()]
            part = sp.part.to_numpy()
            te, fit, val = part == "test", part == "train", part == "val"
            full = ~te
            preds = {} if a.only_logreg else {"knn5": knn_predict(X[full], y[full], X[te], [5], K)[5],
                                              "ncc": ncc_predict(X[full], y[full], X[te], K)}
            if not a.no_logreg:
                sc = lambda yt, pp: metrics_from_confusion(confusion(yt, pp, K))["balanced_accuracy"]  # noqa: E731
                preds["logreg"] = logreg_select_and_fit(X[fit], y[fit], X[val], y[val], X[full], y[full], X[te], K,
                                                        lg["wd_grid"], lg["max_iter"], seed, sc)[0]
            for ev, pr in preds.items():
                recs.append(pd.DataFrame({"image_id": sp.image_id[te].values, "artist": sp["artist"][te].values,
                                          "style": sp["style"][te].values, "protocol": p, "seed": seed,
                                          "evaluator": ev, "y": y[te], "pred": pr}))
    df = pd.concat(recs, ignore_index=True)
    df.to_parquet(out / f"{e}.parquet")
    g = df.assign(c=df.y == df.pred).groupby(["protocol", "evaluator", "seed", "style"]).c.mean() \
        .groupby(["protocol", "evaluator", "seed"]).mean().groupby(["protocol", "evaluator"]).mean().unstack()
    print(f"== {a.dataset}/{e}\n{(100*g).round(1)}", flush=True)
