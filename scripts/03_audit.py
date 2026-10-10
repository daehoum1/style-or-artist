"""Dataset audit on raw and cleaned metadata -> artifacts/reports/<ds>_audit.{json,md}, per-style / style x genre CSV."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, dataset_config  # noqa: E402
from stylebench.data.audit import audit, per_style_table  # noqa: E402
from stylebench.data.base import get_adapter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
a = ap.parse_args()
ad = get_adapter(dataset_config(a.dataset))
raw = ad.load()
cl = pd.read_parquet(ARTIFACTS / "metadata" / a.dataset / "cleaned.parquet")
rep = {"dataset": a.dataset, "raw": audit(raw, ad.has_genre), "cleaned": audit(cl, ad.has_genre),
       "cleaning": json.load(open(ARTIFACTS / "metadata" / a.dataset / "cleaning_stats.json"))}
R = ARTIFACTS / "reports"
json.dump(rep, open(R / f"{a.dataset}_audit.json", "w"), indent=1)
pst = per_style_table(cl, ad.has_genre)
pst.to_csv(R / f"{a.dataset}_per_style.csv")
if ad.has_genre:
    ct = pd.crosstab(cl["style"], cl["genre"].fillna("<missing>"))
    ct.to_csv(R / f"{a.dataset}_style_x_genre.csv")
with open(R / f"{a.dataset}_audit.md", "w") as f:
    f.write(f"# Dataset audit: {a.dataset}\n\n```json\n{json.dumps(rep, indent=1)}\n```\n\n## Per style (cleaned)\n\n")
    f.write(pst.to_markdown() if hasattr(pst, "to_markdown") else pst.to_string())
    if ad.has_genre:
        f.write("\n\n## Style x genre (cleaned)\n\n```\n" + ct.to_string() + "\n```\n")
print(json.dumps(rep["cleaned"], indent=1))
print(pst.to_string())
