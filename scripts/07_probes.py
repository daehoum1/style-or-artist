"""Leakage probes for (dataset, encoders). Pool = controlled pool of the default split config."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

torch.set_num_threads(4)  # several workers share 24 cores

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, dataset_config, load_yaml, split_config  # noqa: E402
from stylebench.data.base import get_adapter  # noqa: E402
from stylebench.data.splits import controlled_pool  # noqa: E402
from stylebench.evaluation.leakage import probe_tasks, run_probes  # noqa: E402
from stylebench.features.cache import available, load_embeddings  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("encoders", nargs="+")
a = ap.parse_args()
ecfg = load_yaml("configs/experiments/evaluation.yaml")
pcfg = ecfg["probes"]
has_genre = get_adapter(dataset_config(a.dataset)).has_genre
clean = pd.read_parquet(ARTIFACTS / "metadata" / a.dataset / "cleaned.parquet")
pool, _ = controlled_pool(clean, split_config()["support"], need_genre=has_genre)
tasks = probe_tasks(pool, pcfg, has_genre)
out = ARTIFACTS / "results" / "probes" / a.dataset
out.mkdir(parents=True, exist_ok=True)
json.dump([{"probe": t["probe"], "style": t["style"], "n_rows": int(len(t["rows"]))} for t in tasks],
          open(out / "tasks.json", "w"), indent=1)
for e in a.encoders:
    if not available(a.dataset, e):
        print(f"[missing] {e}")
        continue
    if (out / f"{e}.parquet").exists():
        print(f"[skip] {e}")
        continue
    E, row, _ = load_embeddings(a.dataset, e)
    df = run_probes(E, row, pool, tasks, pcfg, ecfg["evaluators"]["logreg"]["wd_grid"])
    df["encoder"] = e
    df.to_parquet(out / f"{e}.parquet")
    s = df.groupby(["probe", "evaluator"])["balanced_accuracy_norm"].mean().unstack().round(3)
    print(f"== {e}\n{s}", flush=True)
