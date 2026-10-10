"""Style classification on saved splits for one (dataset, encoder)."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from ..config import ARTIFACTS, load_yaml
from ..features.cache import load_embeddings
from .classifiers import knn_predict, logreg_fit_predict, logreg_select_and_fit, ncc_predict
from .metrics import confusion, metrics_from_confusion


def _file_hash(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]


def evaluate_split(E, row, split: pd.DataFrame, ecfg: dict, seed: int, evaluators=("knn5", "knn_sel", "ncc", "logreg")):
    styles = sorted(split["style"].unique())
    sid = {s: i for i, s in enumerate(styles)}
    K = len(styles)
    X = E[split.image_id.map(row).to_numpy()]
    y = split["style"].map(sid).to_numpy()
    part = split.part.to_numpy()
    fit, val, te = part == "train", part == "val", part == "test"
    full = ~te
    res = []

    def record(name, hp, pred, sel=None):
        C = confusion(y[te], pred, K)
        m = metrics_from_confusion(C)
        res.append({"evaluator": name, "hparam": hp, "val_bacc": sel, **{k: v for k, v in m.items()
                                                                        if k != "per_class"},
                    "_per_class": m["per_class"], "_confusion": C, "_styles": styles})

    ev = ecfg["evaluators"]
    # knn5 (fixed) and knn_sel (k on validation)
    grid = ev["knn_sel"]["k_grid"]
    vp = knn_predict(X[fit], y[fit], X[val], grid, K) if val.any() else None
    tp = knn_predict(X[full], y[full], X[te], sorted(set(grid) | {ev["knn5"]["k"]}), K)
    record("knn5", ev["knn5"]["k"], tp[ev["knn5"]["k"]])
    if vp is not None:
        vs = {k: metrics_from_confusion(confusion(y[val], vp[k], K))["balanced_accuracy"] for k in grid}
        kb = max(grid, key=lambda k: (vs[k], -k))
        record("knn_sel", kb, tp[kb], vs[kb])
    # ncc
    record("ncc", None, ncc_predict(X[full], y[full], X[te], K))
    lg = ev["logreg"]
    if "logreg" not in evaluators or seed not in lg.get("seeds", [seed]):
        return res
    sc = lambda yt, pp: metrics_from_confusion(confusion(yt, pp, K))["balanced_accuracy"]  # noqa: E731
    if val.any():
        pt, wb, vb = logreg_select_and_fit(X[fit], y[fit], X[val], y[val], X[full], y[full], X[te], K,
                                           lg["wd_grid"], lg["max_iter"], seed, sc)
    else:
        wb, vb = 1e-3, None
        (pt,) = logreg_fit_predict(X[full], y[full], [X[te]], K, wb, lg["max_iter"], seed)
    record("logreg", wb, pt, vb)
    return res


def run_style_eval(dataset: str, encoder: str, protocols: list[str], split_root: str | None = None,
                   emb_dataset: str | None = None, out_tag: str = "",
                   evaluators=("knn5", "knn_sel", "ncc", "logreg")) -> pd.DataFrame:
    ecfg = load_yaml("configs/experiments/evaluation.yaml")
    E, row, enc_info = load_embeddings(emb_dataset or dataset, encoder)
    sroot = ARTIFACTS / "splits" / (split_root or dataset)
    out = ARTIFACTS / "results" / "style_eval" / ((split_root or dataset) + out_tag)
    out.mkdir(parents=True, exist_ok=True)
    rows, pcs, confs = [], [], {}
    for p in protocols:
        for f in sorted((sroot / p).glob("seed*.parquet")):
            seed = int(f.stem[4:])
            sp = pd.read_parquet(f)
            t0 = time.time()
            for r in evaluate_split(E, row, sp, ecfg, seed, evaluators):
                key = f"{p}|{seed}|{r['evaluator']}"
                confs[key] = r.pop("_confusion")
                pc = r.pop("_per_class")
                st = r.pop("_styles")
                for i, s in enumerate(st):
                    pcs.append({"protocol": p, "seed": seed, "evaluator": r["evaluator"], "style": s,
                                **{k: float(v[i]) for k, v in pc.items()}})
                rows.append({"dataset": dataset, "split_root": split_root or dataset, "encoder": encoder,
                             "protocol": p, "seed": seed, **r, "split_file_sha": _file_hash(f),
                             "n_train": int((sp.part != "test").sum()), "n_test": int((sp.part == "test").sum()),
                             "seconds": round(time.time() - t0, 2)})
    df = pd.DataFrame(rows)
    df["hparam"] = df["hparam"].astype(str)
    df.to_parquet(out / f"{encoder}.parquet")
    pd.DataFrame(pcs).assign(encoder=encoder).to_parquet(out / f"{encoder}_per_class.parquet")
    np.savez_compressed(out / f"{encoder}_confusion.npz", **confs)
    json.dump({"encoder_config": enc_info, "evaluation_config": ecfg}, open(out / f"{encoder}_provenance.json", "w"),
              indent=1, default=str)
    return df
