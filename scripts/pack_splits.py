"""Pack artifacts/splits/ into a compact, lossless form for distribution (splits/ in the repository).
For every split directory <ds>: <ds>.images.parquet (image_id, style, artist, genre; one row per image),
<ds>.rows.parquet (one row per split-file row: key = <protocol>/seed<k>, idx into the image table, part,
cell, role) and <ds>.attrs.json.gz (the .json sidecar of every split file). MANIFEST.json stores the column
schema, the pandas attrs, the SHA-256 of every original split file and of its content. scripts/unpack_splits.py inverts this."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "splits"
DST = ROOT / "splits"
DST.mkdir(exist_ok=True)
manifest = {}
for ds_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
    ds = ds_dir.name
    files = sorted(ds_dir.glob("*/seed*.parquet"))
    if not files:
        continue
    frames, attrs, images = [], {}, {}
    for f in files:
        key = f"{f.parent.name}/{f.stem}"
        d = pd.read_parquet(f)
        assert isinstance(d.index, pd.RangeIndex) and d.index.start == 0
        for r in d[["image_id", "style", "artist", "genre"]].itertuples(index=False):
            old = images.setdefault(r.image_id, (r.style, r.artist, r.genre))
            assert old == (r.style, r.artist, r.genre), (ds, key, r.image_id)
        frames.append((key, d))
        j = f.with_suffix(".json")
        attrs[key] = json.load(open(j)) if j.exists() else None
        manifest[f"{ds}/{key}"] = {"columns": list(d.columns), "dtypes": [str(t) for t in d.dtypes],
                                   "frame_attrs": d.attrs,
                                   "file_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                   "content_hash": str(int(pd.util.hash_pandas_object(d, index=True).sum()))}
    img = pd.DataFrame([(k, *v) for k, v in sorted(images.items())], columns=["image_id", "style", "artist", "genre"])
    pos = {k: i for i, k in enumerate(img.image_id)}
    rows = []
    for key, d in frames:
        r = pd.DataFrame({"key": key, "idx": d.image_id.map(pos).astype("int32"), "part": d.part.astype(str)})
        for c in ["cell", "role"]:
            r[c] = d[c].astype(str) if c in d.columns else ""
        rows.append(r)
    R = pd.concat(rows, ignore_index=True)
    t = pa.Table.from_pandas(R, preserve_index=False)
    pq.write_table(t, DST / f"{ds}.rows.parquet", compression="zstd", compression_level=19,
                   use_dictionary=["key", "part", "cell", "role"], column_encoding={"idx": "DELTA_BINARY_PACKED"})
    img.to_parquet(DST / f"{ds}.images.parquet", compression="zstd", index=False)
    with gzip.open(DST / f"{ds}.attrs.json.gz", "wt") as fh:
        json.dump(attrs, fh, default=str)
    print(ds, len(files), "files", len(img), "images", flush=True)
json.dump(manifest, open(DST / "MANIFEST.json", "w"), indent=0)
