"""Restore artifacts/splits/<ds>/<protocol>/seed<k>.parquet (+ .json) from the packed files in splits/
and verify each restored file against MANIFEST.json (content hash; file SHA-256 additionally matches with
the pinned pandas/pyarrow versions of requirements.txt)."""
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "splits"
DST = ROOT / "artifacts" / "splits"
man = json.load(open(SRC / "MANIFEST.json"))
n = bad = sha_ok = 0
for rows_f in sorted(SRC.glob("*.rows.parquet")):
    ds = rows_f.name[: -len(".rows.parquet")]
    img = pd.read_parquet(SRC / f"{ds}.images.parquet")
    R = pd.read_parquet(rows_f)
    attrs = json.load(gzip.open(SRC / f"{ds}.attrs.json.gz", "rt"))
    for key, r in R.groupby("key", sort=False, observed=True):
        m = man[f"{ds}/{key}"]
        sel = img.iloc[r.idx.to_numpy()].reset_index(drop=True)
        d = pd.DataFrame({"image_id": sel.image_id.astype(object)})
        for c in ["style", "artist", "genre"]:
            d[c] = sel[c].astype("string")
        for c in ["part", "cell", "role"]:
            if c in m["columns"]:
                d[c] = r[c].astype(str).to_numpy().astype(object)
        d = d[m["columns"]]
        d.attrs = m.get("frame_attrs", {})
        out = DST / ds / f"{key}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(out)
        if attrs.get(key) is not None:
            json.dump(attrs[key], open(out.with_suffix(".json"), "w"), indent=1, default=str)
        ok = str(int(pd.util.hash_pandas_object(d, index=True).sum())) == m["content_hash"]
        sha_ok += hashlib.sha256(out.read_bytes()).hexdigest() == m["file_sha256"]
        n += 1
        bad += not ok
        if not ok:
            print("CONTENT MISMATCH", ds, key)
print(f"restored {n} split files; content verified for {n - bad}; byte-identical {sha_ok}")
