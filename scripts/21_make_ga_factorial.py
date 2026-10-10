"""Revision round 2 (D1): matched and exposure-matched factorial splits on a GENRE-AGNOSTIC partition:
artists from the Artist-D partition of the seed (chosen without regard to genre) x held-out genres of the
Genre-D/Joint split of the same seed. All leakage assertions are applied.
Output: artifacts/splits/<ds>_ga/{mf_*,emf_*}/seed<k>.parquet (+ json); artifacts/reports/<ds>_ga_report.csv"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, load_yaml, split_config  # noqa: E402
from stylebench.data.splits import (MF_PROTOCOLS, controlled_pool, split_exposure_matched,  # noqa: E402
                                    split_matched_factorial, validate_exposure_matched)
from stylebench.data.validate import validate_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
a = ap.parse_args()
cfg = split_config()
src = ARTIFACTS / "splits" / a.dataset
dst = ARTIFACTS / "splits" / f"{a.dataset}_ga"
clean = pd.read_parquet(ARTIFACTS / "metadata" / a.dataset / "cleaned.parquet")
hashes = pd.read_parquet(ARTIFACTS / "metadata" / a.dataset / "hashes.parquet")
maxh = load_yaml("configs/cleaning.yaml")["near_duplicate_max_hamming"]
pool, _ = controlled_pool(clean, cfg["support"])
rows = []
for seed in cfg["seeds"]:
    A_te = set(json.load(open(src / "artist_disjoint" / f"seed{seed}.json"))["test_artists"])
    G_te = list(json.load(open(src / "joint" / f"seed{seed}.json"))["test_genres"])
    styles = sorted(pd.read_parquet(src / "joint" / f"seed{seed}.parquet")["style"].unique())
    p = pool[pool["style"].isin(styles)]
    mf = split_matched_factorial(p, A_te, G_te, styles, cfg, seed)
    for c, d in mf.items():
        validate_split(d, c, cfg["support"], hashes, maxh)
    ref = mf["mf_iid"]
    for c in MF_PROTOCOLS:
        d = mf[c]
        assert set(d[d.part == "test"].image_id) == set(ref[ref.part == "test"].image_id)
        assert (d[d.part != "test"].groupby("style").size() == ref[ref.part != "test"].groupby("style").size()).all()
    emf = split_exposure_matched(p, A_te, G_te, styles, cfg, seed)
    res = validate_exposure_matched(emf, A_te, G_te)
    for c, d in {**mf, **emf}.items():
        (dst / c).mkdir(parents=True, exist_ok=True)
        cols = ["image_id", "style", "artist", "genre", "part"] + [x for x in ["cell", "role"] if x in d.columns]
        d[cols].to_parquet(dst / c / f"seed{seed}.parquet")
        json.dump({k: v for k, v in d.attrs.items()}, open(dst / c / f"seed{seed}.json", "w"), indent=1, default=str)
    rows.append({"seed": seed, "mf_test": int((ref.part == "test").sum()), "mf_styles": ref["style"].nunique(),
                 "mf_dropped": json.dumps(ref.attrs.get("mf_dropped", {})), "emf_test": res["n_test"],
                 "emf_test_artists": res["n_test_artists"], "emf_addon": res["n_addon"],
                 "emf_styles": len(emf["emf_AG"].attrs["emf_style_stats"])})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ARTIFACTS / "reports" / f"{a.dataset}_ga_report.csv", index=False)
