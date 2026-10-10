"""Generate, validate and save all split protocols for every seed.
Output: artifacts/splits/<ds>/<protocol>/seed<k>.parquet (+ .json diagnostics), pool / core-style tables,
and the split-validation report artifacts/reports/<ds>_split_report.md"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, dataset_config, load_yaml, split_config  # noqa: E402
from stylebench.data.base import get_adapter  # noqa: E402
from stylebench.data.splits import controlled_pool, genre_holdout_candidates, make_seed_splits  # noqa: E402
from stylebench.data.validate import validate_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--config", default="default")
ap.add_argument("--tag", default="", help="suffix for sensitivity variants")
a = ap.parse_args()
cfg = split_config(a.config)
has_genre = get_adapter(dataset_config(a.dataset)).has_genre
meta = ARTIFACTS / "metadata" / a.dataset
clean = pd.read_parquet(meta / "cleaned.parquet")
hashes = pd.read_parquet(meta / "hashes.parquet")
maxh = load_yaml("configs/cleaning.yaml")["near_duplicate_max_hamming"]
name = a.dataset + (f"_{a.tag}" if a.tag else "")
out = ARTIFACTS / "splits" / name
out.mkdir(parents=True, exist_ok=True)

pool, style_tab = controlled_pool(clean, cfg["support"], need_genre=has_genre)
style_tab.to_csv(out / "core_style_table.csv")
cands, removed = genre_holdout_candidates(pool, cfg["support"], cfg["genre_holdout"]) if has_genre else ([None], {})
if removed:  # recorded in the core style table; these styles leave the pool for every protocol
    style_tab.loc[list(removed), "excluded_reason"] = [removed[s] for s in removed]
    style_tab.to_csv(out / "core_style_table.csv")
    pool = pool[~pool["style"].isin(removed)].reset_index(drop=True)
if has_genre and not cands:
    raise SystemExit("no feasible held-out genre set; relax genre_holdout settings (recorded)")
json.dump({"n_candidates": len(cands), "removed_styles": removed, "candidates": cands[:2000]}, open(out / "genre_holdout_candidates.json", "w"),
          indent=0)
print(f"pool={len(pool)} core styles={pool['style'].nunique()} genre candidates={len(cands)}")

rows, checks = [], []
for seed in cfg["seeds"]:
    G = cands[np.random.default_rng([seed, 7]).integers(len(cands))] if has_genre else None
    sp = make_seed_splits(pool, cfg, seed, G, has_genre)
    for p, d in sp.items():
        res = validate_split(d, p, cfg["support"], hashes, maxh)   # raises on any leakage
        (out / p).mkdir(exist_ok=True)
        cols = ["image_id", "style", "artist", "genre", "part"] + (["cell"] if "cell" in d.columns else [])
        d[cols].to_parquet(out / p / f"seed{seed}.parquet")
        diag = {k: v for k, v in d.attrs.items()}
        diag["checks"] = res
        side, te = d[d.part != "test"], d[d.part == "test"]
        diag["per_style"] = {
            s: {"train": int((side["style"] == s).sum()), "val": int(((d.part == "val") & (d["style"] == s)).sum()),
                "test": int((te["style"] == s).sum()),
                "train_artists": int(side[side["style"] == s]["artist"].nunique()),
                "test_artists": int(te[te["style"] == s]["artist"].nunique()),
                "train_genres": int(side[side["style"] == s]["genre"].nunique()),
                "test_genres": int(te[te["style"] == s]["genre"].nunique())}
            for s in sorted(d["style"].unique())}
        json.dump(diag, open(out / p / f"seed{seed}.json", "w"), indent=1, default=str)
        rows.append({"seed": seed, "protocol": p, "n_train_fit": int((d.part == "train").sum()),
                     "n_val": int((d.part == "val").sum()),
                     "n_sel_excluded": int((d.part == "train_selection_excluded").sum()),
                     "n_train_total": int((d.part != "test").sum()), "n_test": int((d.part == "test").sum()),
                     "n_styles": d["style"].nunique(), "artist_overlap": res["artist_overlap_count"],
                     "genre_overlap": res["genre_overlap_count"],
                     "near_dup_cross": res.get("near_duplicate_cross_pairs"),
                     "test_genres": ",".join(diag.get("test_genres", [])),
                     "n_test_artists": te["artist"].nunique(),
                     "n_dropped_styles": len(diag["dropped_styles"]),
                     "all_checks_pass": all(v for k, v in res.items() if isinstance(v, bool))})
    if "mf_iid" in sp:  # factorial consistency: one test set, identical per-style training sizes
        ref = sp["mf_iid"]
        for p in ["mf_artist", "mf_genre", "mf_joint"]:
            d = sp[p]
            assert set(d[d.part == "test"].image_id) == set(ref[ref.part == "test"].image_id), "MF test differs"
            assert (d[d.part != "test"].groupby("style").size() ==
                    ref[ref.part != "test"].groupby("style").size()).all(), "MF train sizes differ"
    print(f"seed {seed}: G_te={G} dropped={list(sp['iid'].attrs['dropped_styles'])}")
summ = pd.DataFrame(rows)
summ.to_csv(out / "split_summary.csv", index=False)
print(summ.to_string())
