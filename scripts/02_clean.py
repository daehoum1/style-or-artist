"""Clean a dataset. Outputs (artifacts/metadata/<ds>/):
cleaned.parquet, cleaning_decisions.csv, cleaning_stats.json, label_map_<field>.csv"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, dataset_config, load_yaml  # noqa: E402
from stylebench.data.base import get_adapter  # noqa: E402
from stylebench.data.cleaning import clean  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
a = ap.parse_args()
out = ARTIFACTS / "metadata" / a.dataset
raw = get_adapter(dataset_config(a.dataset)).load()
h = pd.read_parquet(out / "hashes.parquet")
cfg = load_yaml("configs/cleaning.yaml")
cleaned, dec, stats = clean(raw, h, cfg["near_duplicate_max_hamming"])
cleaned.to_parquet(out / "cleaned.parquet")
dec.to_csv(out / "cleaning_decisions.csv", index=False)
for field in ["style", "artist", "genre"]:
    rawcol = {"wikiart": {"style": "raw_style_folder", "artist": "raw_artist_slug", "genre": "raw_genre_id"}}.get(
        a.dataset, {}).get(field, f"raw_{field}")
    if rawcol in raw.columns:
        m = raw.groupby([rawcol, field], dropna=False).size().reset_index(name="n_images")
        m.columns = ["raw", "normalized", "n_images"]
        m.to_csv(out / f"label_map_{field}.csv", index=False)
stats["reason_counts"] = dec.reason.value_counts().to_dict()
json.dump(stats, open(out / "cleaning_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
