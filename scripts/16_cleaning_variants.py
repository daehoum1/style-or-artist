"""Revision: cleaning-policy sensitivity. Builds two variants per dataset
  <ds>_phash8   : near-duplicate threshold Hamming <= 8 (main: 4)
  <ds>_keepconf : duplicate groups with conflicting styles keep one copy with its own label (main: drop group)
writes artifacts/metadata/<variant>/, configs/datasets/<variant>.yaml, extracts embeddings of images that the
main cleaning had dropped, and assembles artifacts/embeddings/<variant>/<encoder>/."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stylebench.config import ARTIFACTS, dataset_config  # noqa: E402
from stylebench.data.base import get_adapter  # noqa: E402
from stylebench.data.cleaning import clean  # noqa: E402
from stylebench.features.extract import extract  # noqa: E402
from stylebench.models.base import encoder_configs  # noqa: E402

for ds in ["wikiart", "multitask_painting"]:
    raw = get_adapter(dataset_config(ds)).load()
    h = pd.read_parquet(ARTIFACTS / "metadata" / ds / "hashes.parquet")
    for var, mh, pol in [("phash8", 8, "drop"), ("keepconf", 4, "keep_representative")]:
        name = f"{ds}_{var}"
        out = ARTIFACTS / "metadata" / name
        out.mkdir(parents=True, exist_ok=True)
        c, dec, stats = clean(raw, h, mh, pol)
        c.to_parquet(out / "cleaned.parquet")
        dec.to_csv(out / "cleaning_decisions.csv", index=False)
        stats["reason_counts"] = dec.reason.value_counts().to_dict()
        json.dump(stats, open(out / "cleaning_stats.json", "w"), indent=1)
        shutil.copy(ARTIFACTS / "metadata" / ds / "hashes.parquet", out / "hashes.parquet")
        cfg = dataset_config(ds)
        cfg["name"] = name
        yaml.safe_dump(cfg, open(ROOT / "configs" / "datasets" / f"{name}.yaml", "w"))
        base_ids = set(pd.read_parquet(ARTIFACTS / "metadata" / ds / "cleaned.parquet").image_id)
        missing = c[~c.image_id.isin(base_ids)][["image_id", "path"]].reset_index(drop=True)
        print(name, "images", len(c), "not in main cleaned set (to embed):", len(missing), flush=True)
        for e in encoder_configs():
            tgt = ARTIFACTS / "embeddings" / name / e
            if (tgt / "embeddings.npy").exists():
                continue
            bE = np.load(ARTIFACTS / "embeddings" / ds / e / "embeddings.npy")
            bm = pd.read_parquet(ARTIFACTS / "embeddings" / ds / e / "metadata.parquet")
            brow = dict(zip(bm.image_id, bm.row))
            parts, ids = [], []
            keep = c[c.image_id.isin(base_ids)].image_id.tolist()
            parts.append(bE[[brow[i] for i in keep]])
            ids += keep
            if len(missing):
                extract(f"{name}_extra", e, missing, 256, 16)
                xe = np.load(ARTIFACTS / "embeddings" / f"{name}_extra" / e / "embeddings.npy")
                parts.append(xe)
                ids += missing.image_id.tolist()
            tgt.mkdir(parents=True, exist_ok=True)
            np.save(tgt / "embeddings.npy", np.concatenate(parts).astype(np.float32))
            pd.DataFrame({"image_id": ids, "row": np.arange(len(ids))}).to_parquet(tgt / "metadata.parquet")
            shutil.copy(ARTIFACTS / "embeddings" / ds / e / "config.json", tgt / "config.json")
