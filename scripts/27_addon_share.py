"""Revision round 3 (R1-C2): sensitivity of the exposure-matched factorial to the share of add-on images.
The add-on images are the only images that differ between the four exposure-matched conditions (on average
about 8% of the training side). Here the shared base set B_s of every style is subsampled to a fraction f of its
size (the same subsample in all four conditions), while the test set and the add-on images are unchanged, so the
add-on share, and with it the difference in genre composition between conditions, grows as f decreases.
Output: artifacts/splits/<ds>_ao<100f>/emf_*/seed<k>.parquet (+ json), embeddings symlink,
        artifacts/reports/<ds>_addon_share_report.csv"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, split_config  # noqa: E402
from stylebench.data.splits import EMF_PROTOCOLS, validate_exposure_matched  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--fractions", default="0.5,0.25,0.1")
a = ap.parse_args()
cfg = split_config()
src = ARTIFACTS / "splits" / a.dataset
rows = []
for f in [float(x) for x in a.fractions.split(",")]:
    tag = f"{a.dataset}_ao{int(round(100 * f))}"
    dst = ARTIFACTS / "splits" / tag
    emb = ARTIFACTS / "embeddings" / tag
    if not emb.exists():
        os.symlink(a.dataset, emb)
    for seed in cfg["seeds"]:
        sp = {c: pd.read_parquet(src / c / f"seed{seed}.parquet") for c in EMF_PROTOCOLS}
        meta = {c: json.load(open(src / c / f"seed{seed}.json")) for c in EMF_PROTOCOLS}
        base = sp["emf_AG"][sp["emf_AG"].role == "base"]
        keep = []
        for si, (s, g) in enumerate(base.groupby("style", sort=True)):
            rng = np.random.default_rng([seed, 2727, int(round(1000 * f)), si])
            ids = np.sort(g.image_id.to_numpy())
            keep += list(rng.choice(ids, max(1, int(round(f * len(ids)))), replace=False))
        keep = set(keep)
        out = {c: d[(d.role != "base") | d.image_id.isin(keep)].reset_index(drop=True) for c, d in sp.items()}
        G_te = meta["emf_AG"]["test_genres"]
        te = out["emf_AG"][out["emf_AG"].part == "test"]
        res = validate_exposure_matched(out, set(te["artist"]), G_te)
        for c, d in out.items():
            (dst / c).mkdir(parents=True, exist_ok=True)
            d.to_parquet(dst / c / f"seed{seed}.parquet")
            json.dump({**meta[c], "base_fraction": f}, open(dst / c / f"seed{seed}.json", "w"), indent=1, default=str)
        tg = set(G_te)
        share = {c: float(d[d.part != "test"]["genre"].isin(tg).mean()) for c, d in out.items()}
        rows.append({"dataset": a.dataset, "base_fraction": f, "seed": seed, **res,
                     "addon_share": res["n_addon"] / res["n_train"],
                     **{f"test_genre_share_{c}": v for c, v in share.items()}})
        print(rows[-1], flush=True)
for seed in cfg["seeds"]:
    sp = {c: pd.read_parquet(src / c / f"seed{seed}.parquet") for c in EMF_PROTOCOLS}
    G_te = json.load(open(src / "emf_AG" / f"seed{seed}.json"))["test_genres"]
    te = sp["emf_AG"][sp["emf_AG"].part == "test"]
    res = validate_exposure_matched(sp, set(te["artist"]), G_te)
    share = {c: float(d[d.part != "test"]["genre"].isin(set(G_te)).mean()) for c, d in sp.items()}
    rows.append({"dataset": a.dataset, "base_fraction": 1.0, "seed": seed, **res,
                 "addon_share": res["n_addon"] / res["n_train"], **{f"test_genre_share_{c}": v for c, v in share.items()}})
pd.DataFrame(rows).to_csv(ARTIFACTS / "reports" / f"{a.dataset}_addon_share_report.csv", index=False)
