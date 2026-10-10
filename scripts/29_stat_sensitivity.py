"""Revision round 3 (R2-C4): statistical sensitivity of the exposure-matched effects (kNN-5).
  * artist-macro balanced accuracy (mean over test artists of per-artist accuracy) instead of style-macro,
  * a global artist-cluster bootstrap: the same artist appears in the test sets of several seeds, so artists are
    resampled once from the union over seeds and every seed's test images of a drawn artist enter with the
    artist's multiplicity (style-balanced accuracy, styles losing all test images keep their point recall),
  * percentile intervals at the per-encoder Bonferroni level (1 - 0.05/13) for the global bootstrap,
  * test-artist overlap between seeds (mean pairwise Jaccard index).
Output: artifacts/final/rev3/stat_sensitivity.csv, seed_overlap.csv"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS  # noqa: E402

OUT = ARTIFACTS / "final" / "rev3"
OUT.mkdir(parents=True, exist_ok=True)
C = ["emf_AG", "emf_A", "emf_G", "emf_none"]
R = 2000
NENC = 13
rng = np.random.default_rng(3)


def eff(Y):
    return {"artist": Y["emf_AG"] - Y["emf_G"], "genre_Aplus": Y["emf_AG"] - Y["emf_A"],
            "genre_Aminus": Y["emf_G"] - Y["emf_none"],
            "interaction": (Y["emf_AG"] - Y["emf_A"]) - (Y["emf_G"] - Y["emf_none"])}


rows, ov = [], []
for ds in ["wikiart", "multitask_painting", "mp_novel", "wikiart_ga", "multitask_painting_ga"]:
    pdir = ARTIFACTS / "results" / "preds" / ds
    if not pdir.exists():
        continue
    first = True
    for f in sorted(pdir.glob("*.parquet")):
        P = pd.read_parquet(f)
        P = P[(P.evaluator == "knn5") & P.protocol.isin(C)].copy()
        P["ok"] = (P.y == P.pred).astype(float)
        seeds = sorted(P.seed.unique())
        if first:
            sets = {s: set(P[(P.seed == s) & (P.protocol == "emf_AG")].artist) for s in seeds}
            j = [len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in itertools.combinations(seeds, 2)]
            cnt = pd.Series([x for s in seeds for x in sets[s]]).value_counts()
            ov.append({"dataset": ds, "mean_pairwise_jaccard": np.mean(j), "artists_union": len(cnt),
                       "mean_seeds_per_artist": cnt.mean(), "share_artists_in_one_seed": (cnt == 1).mean()})
            first = False
        # artist-macro point effects (mean over seeds)
        am = {c: P[P.protocol == c].groupby(["seed", "artist"]).ok.mean().groupby("seed").mean() for c in C}
        sm = {c: P[P.protocol == c].groupby(["seed", "style"]).ok.mean().groupby("seed").mean() for c in C}
        e_am, e_sm = eff(am), eff(sm)
        # global artist-cluster bootstrap (style-macro within seed, mean over seeds)
        arts = sorted(P.artist.unique())
        ai = {a: i for i, a in enumerate(arts)}
        W = rng.multinomial(len(arts), np.ones(len(arts)) / len(arts), size=R).astype(float)
        boot = {c: np.zeros((R, len(seeds))) for c in C}
        boot_am = {c: np.zeros((R, len(seeds))) for c in C}
        for si, s in enumerate(seeds):
            q = P[P.seed == s]
            stys = sorted(q["style"].unique())
            st = {x: i for i, x in enumerate(stys)}
            base = q[q.protocol == "emf_AG"]
            N = np.zeros((len(arts), len(stys)))
            np.add.at(N, (base.artist.map(ai).to_numpy(), base["style"].map(st).to_numpy()), 1)
            WN = W @ N
            for c in C:
                d = q[q.protocol == c]
                K = np.zeros_like(N)
                np.add.at(K, (d.artist.map(ai).to_numpy(), d["style"].map(st).to_numpy()), d.ok.to_numpy())
                rec = K.sum(0) / N.sum(0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    boot[c][:, si] = np.where(WN > 0, (W @ K) / WN, rec[None, :]).mean(1)
                n_a, k_a = N.sum(1), K.sum(1)
                pres = n_a > 0
                Wp = W[:, pres]
                tot = Wp.sum(1)
                boot_am[c][:, si] = np.where(tot > 0, Wp @ (k_a[pres] / n_a[pres]) / np.maximum(tot, 1),
                                             (k_a[pres] / n_a[pres]).mean())
        gb = eff({c: v.mean(1) for c, v in boot.items()})
        gba = eff({c: v.mean(1) for c, v in boot_am.items()})
        a_b = 0.05 / NENC
        for k in e_sm:
            rows.append({"dataset": ds, "encoder": f.stem, "effect": k, "style_macro": e_sm[k].mean(),
                         "artist_macro": e_am[k].mean(),
                         "glob_lo": np.quantile(gb[k], .025), "glob_hi": np.quantile(gb[k], .975),
                         "glob_bonf_lo": np.quantile(gb[k], a_b / 2), "glob_bonf_hi": np.quantile(gb[k], 1 - a_b / 2),
                         "am_lo": np.quantile(gba[k], .025), "am_hi": np.quantile(gba[k], .975)})
        print(ds, f.stem, flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / "stat_sensitivity.csv", index=False)
pd.DataFrame(ov).to_csv(OUT / "seed_overlap.csv", index=False)
print(pd.DataFrame(ov).round(3))
g = res.groupby(["dataset", "effect"])
print(pd.DataFrame({"style": 100 * g.style_macro.mean(), "artist": 100 * g.artist_macro.mean(),
                    "glob_pos": g.glob_lo.apply(lambda x: (x > 0).sum()),
                    "bonf_pos": g.glob_bonf_lo.apply(lambda x: (x > 0).sum()),
                    "am_pos": g.am_lo.apply(lambda x: (x > 0).sum()),
                    "glob_width": 100 * (g.glob_hi.mean() - g.glob_lo.mean())}).round(2))
