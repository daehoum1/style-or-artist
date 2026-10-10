"""Revision: exposure-matched factorial splits from the saved Joint partitions (same seeds).
Output: artifacts/splits/<ds>/emf_*/seed<k>.parquet + .json; artifacts/reports/<ds>_emf_report.csv"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, split_config  # noqa: E402
from stylebench.data.splits import controlled_pool, split_exposure_matched, validate_exposure_matched  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
a = ap.parse_args()
cfg = split_config()
root = ARTIFACTS / "splits" / a.dataset
clean = pd.read_parquet(ARTIFACTS / "metadata" / a.dataset / "cleaned.parquet")
pool, _ = controlled_pool(clean, cfg["support"])
rows = []
for seed in cfg["seeds"]:
    jd = json.load(open(root / "joint" / f"seed{seed}.json"))
    A_te, G_te = set(jd["test_artists"]), list(jd["test_genres"])
    jp = pd.read_parquet(root / "joint" / f"seed{seed}.parquet")
    styles = sorted(jp["style"].unique())
    sp = split_exposure_matched(pool[pool["style"].isin(styles)], A_te, G_te, styles, cfg, seed)
    res = validate_exposure_matched(sp, A_te, G_te)
    for c, d in sp.items():
        (root / c).mkdir(exist_ok=True)
        d[["image_id", "style", "artist", "genre", "part", "cell", "role"]].to_parquet(root / c / f"seed{seed}.parquet")
        json.dump({k: v for k, v in d.attrs.items()}, open(root / c / f"seed{seed}.json", "w"), indent=1, default=str)
    st = pd.DataFrame(sp["emf_AG"].attrs["emf_style_stats"])
    rows.append({"seed": seed, **res, "n_styles": len(st), "excluded": json.dumps(sp["emf_AG"].attrs["emf_excluded"]),
                 "emf_G_genre_matched_share": float((st.emf_G_genre_matched * st.k).sum() / max(st.k.sum(), 1))})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ARTIFACTS / "reports" / f"{a.dataset}_emf_report.csv", index=False)
