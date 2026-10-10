"""Revision round 2 (D2): zero-shot style classification with the text towers of the vision-language encoders
(prompt "a painting in the style of {style}"), evaluated on the IID, Artist-D, Genre-D and Joint test sets of every
seed. No training painting is used, so differences between test sets reflect test-set composition only.
Output: artifacts/results/rev2/zero_shot.csv"""
import sys
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS  # noqa: E402
from stylebench.evaluation.metrics import confusion, metrics_from_confusion  # noqa: E402
from stylebench.features.cache import load_embeddings  # noqa: E402
from stylebench.models.base import encoder_configs  # noqa: E402

TEMPLATE = "a painting in the style of {}"
cfgs = encoder_configs()
rows = []
for e, c in cfgs.items():
    if c["lib"] != "open_clip":
        continue
    if c["arch"].startswith("hf-hub:"):
        model, _, _ = open_clip.create_model_and_transforms(c["arch"])
    else:
        model, _, _ = open_clip.create_model_and_transforms(c["arch"], pretrained=c["pretrained"])
    tok = open_clip.get_tokenizer(c["arch"])
    model = model.cuda().eval()
    for d in ["wikiart", "multitask_painting"]:
        E, row, _ = load_embeddings(d, e)
        for p in ["iid", "artist_disjoint", "genre_disjoint", "joint"]:
            for f in sorted((ARTIFACTS / "splits" / d / p).glob("seed*.parquet")):
                sp = pd.read_parquet(f)
                styles = sorted(sp["style"].unique())
                with torch.no_grad():
                    T = model.encode_text(tok([TEMPLATE.format(s) for s in styles]).cuda()).float()
                T = torch.nn.functional.normalize(T, dim=1).cpu().numpy()
                te = sp[sp.part == "test"]
                X = E[te.image_id.map(row).to_numpy()]
                pred = (X @ T.T).argmax(1)
                y = te["style"].map({s: i for i, s in enumerate(styles)}).to_numpy()
                m = metrics_from_confusion(confusion(y, pred, len(styles)))
                rows.append({"dataset": d, "encoder": e, "protocol": p, "seed": int(f.stem[4:]),
                             "bacc": m["balanced_accuracy"], "acc": m["accuracy"]})
    print(e, flush=True)
    del model
    torch.cuda.empty_cache()
out = ARTIFACTS / "results" / "rev2"
df = pd.DataFrame(rows)
df.to_csv(out / "zero_shot.csv", index=False)
print((100 * df.groupby(["dataset", "encoder", "protocol"]).bacc.mean().unstack()).round(1).to_string())
