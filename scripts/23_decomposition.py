"""Revision round 2 (C5): training-size control and decomposition of the Artist-D -> Joint drop (kNN-5 bAcc).
(a) IID training sets subsampled per style to the Genre-D and Joint training sizes of the same seed, and Artist-D
    training sets subsampled to the Joint sizes; evaluated on their own test sets.
(b) Decomposition on the Joint partition of each seed (test set = TT = the Joint test set):
      A = standard Artist-D;  B = train on BB u AT (artist-disjoint, genres seen), test TT;
      C = B with per-style training size subsampled to |BB_s|;  D = standard Joint (train BB, test TT).
    A-D = (A-B) test set/partition + (B-C) training size + (C-D) genre exposure.
Output: artifacts/results/rev2/decomposition.csv, size_control.csv"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS, split_config  # noqa: E402
from stylebench.data.splits import controlled_pool  # noqa: E402
from stylebench.evaluation.classifiers import knn_predict  # noqa: E402
from stylebench.evaluation.metrics import confusion, metrics_from_confusion  # noqa: E402
from stylebench.features.cache import load_embeddings  # noqa: E402
from stylebench.models.base import encoder_configs  # noqa: E402

SP = ARTIFACTS / "splits"


def bacc(E, row, tr, te, styles):
    sid = {s: i for i, s in enumerate(styles)}
    Xtr, Xte = E[tr.image_id.map(row).to_numpy()], E[te.image_id.map(row).to_numpy()]
    ytr, yte = tr["style"].map(sid).to_numpy(), te["style"].map(sid).to_numpy()
    p = knn_predict(Xtr, ytr, Xte, [5], len(styles))[5]
    return metrics_from_confusion(confusion(yte, p, len(styles)))["balanced_accuracy"]


def subsample(df, sizes, rng):
    parts = []
    for s, g in df.groupby("style"):
        n = min(int(sizes.get(s, 0)), len(g))
        parts.append(g.iloc[rng.choice(len(g), n, replace=False)])
    return pd.concat(parts)


dec, size = [], []
for d in ["wikiart", "multitask_painting"]:
    pool, _ = controlled_pool(pd.read_parquet(ARTIFACTS / "metadata" / d / "cleaned.parquet"), split_config()["support"])
    encs = list(encoder_configs())
    EMB = {e: load_embeddings(d, e)[:2] for e in encs}
    for seed in range(10):
        rng = np.random.default_rng([seed, 2323])
        sp = {p: pd.read_parquet(SP / d / p / f"seed{seed}.parquet") for p in ["iid", "artist_disjoint", "genre_disjoint", "joint"]}
        styles = sorted(sp["joint"]["style"].unique())
        side = {p: v[v.part != "test"] for p, v in sp.items()}
        test = {p: v[v.part == "test"] for p, v in sp.items()}
        sz = {p: side[p].groupby("style").size() for p in side}
        jd = json.load(open(SP / d / "joint" / f"seed{seed}.json"))
        pp = pool[pool["style"].isin(styles)]
        am, gm = pp["artist"].isin(jd["test_artists"]), pp["genre"].isin(jd["test_genres"])
        BBAT = pp[~am]
        assert set(side["joint"].image_id) == set(pp[~am & ~gm].image_id)
        BBAT_c = subsample(BBAT, sz["joint"], rng)
        sub = {"iid@genre": subsample(side["iid"], sz["genre_disjoint"], rng),
               "iid@joint": subsample(side["iid"], sz["joint"], rng),
               "artist@joint": subsample(side["artist_disjoint"], sz["joint"], rng)}
        for e in encs:
            E, row = EMB[e]
            A = bacc(E, row, side["artist_disjoint"], test["artist_disjoint"], styles)
            B = bacc(E, row, BBAT, test["joint"], styles)
            C = bacc(E, row, BBAT_c, test["joint"], styles)
            D = bacc(E, row, side["joint"], test["joint"], styles)
            dec.append({"dataset": d, "seed": seed, "encoder": e, "A_artistD": A, "B_jointpart_full": B,
                        "C_jointpart_sizematched": C, "D_joint": D})
            size.append({"dataset": d, "seed": seed, "encoder": e,
                         "iid": bacc(E, row, side["iid"], test["iid"], styles),
                         "iid_at_genre_size": bacc(E, row, sub["iid@genre"], test["iid"], styles),
                         "iid_at_joint_size": bacc(E, row, sub["iid@joint"], test["iid"], styles),
                         "artist": A, "artist_at_joint_size": bacc(E, row, sub["artist@joint"], test["artist_disjoint"], styles),
                         "genre": bacc(E, row, side["genre_disjoint"], test["genre_disjoint"], styles), "joint": D})
        print(d, seed, flush=True)
out = ARTIFACTS / "results" / "rev2"
X = pd.DataFrame(dec)
X.to_csv(out / "decomposition.csv", index=False)
Z = pd.DataFrame(size)
Z.to_csv(out / "size_control.csv", index=False)
m = X.groupby("dataset").mean(numeric_only=True) * 100
m["testset_partition"] = m.A_artistD - m.B_jointpart_full
m["train_size"] = m.B_jointpart_full - m.C_jointpart_sizematched
m["genre_exposure"] = m.C_jointpart_sizematched - m.D_joint
m["total"] = m.A_artistD - m.D_joint
print(m.drop(columns="seed").round(2).to_string())
print((Z.groupby("dataset").mean(numeric_only=True) * 100).drop(columns="seed").round(2).to_string())
