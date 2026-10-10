"""Analysis of the factorial designs from saved per-image predictions (revision rounds 1-2).
Designs: matched (mf_*) and exposure-matched (emf_*); datasets: main, mp_novel, genre-agnostic partitions (*_ga).
For each dataset x encoder x evaluator x design:
  * effects (artist, genre|A+, genre|A-, interaction) as means over seeds (balanced accuracy),
  * the same effects on the logit scale and as relative changes (C1),
  * intervals: seed bootstrap; hierarchical bootstrap (seeds, and test artists as clusters within seeds) with
    percentile and BCa (bias-corrected and accelerated; acceleration from a leave-one-(seed,artist)-out
    jackknife) intervals, bootstrap bias, share of replicates in which some style lost all test images and
    the percentile interval restricted to replicates without such loss (C4),
  * optional exclusion of flagged test images (C6: --exclude <csv with dataset,seed,image_id>).
Outputs in artifacts/final/factorial/ (or factorial_<tag>/)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exclude", default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--datasets", default="wikiart,multitask_painting,mp_novel,wikiart_ga,multitask_painting_ga")
a = ap.parse_args()
OUT = ARTIFACTS / "final" / ("factorial" + (f"_{a.tag}" if a.tag else ""))
OUT.mkdir(parents=True, exist_ok=True)
DESIGNS = {"mf": {"AG": "mf_iid", "A": "mf_genre", "G": "mf_artist", "none": "mf_joint"},
           "emf": {"AG": "emf_AG", "A": "emf_A", "G": "emf_G", "none": "emf_none"}}
R = 2000
DELTA = 0.05
EXCL = None
if a.exclude:
    EXCL = pd.read_csv(a.exclude)
    EXCL = set(zip(EXCL.dataset, EXCL.seed, EXCL.image_id))
rng = np.random.default_rng(0)


def effects(Y):
    return {"artist": Y["AG"] - Y["G"], "genre_Aplus": Y["AG"] - Y["A"], "genre_Aminus": Y["G"] - Y["none"],
            "interaction": (Y["AG"] - Y["A"]) - (Y["G"] - Y["none"])}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def rel(Y):  # relative changes (fraction of the reference condition)
    return {"artist": (Y["AG"] - Y["G"]) / Y["AG"], "genre_Aplus": (Y["AG"] - Y["A"]) / Y["AG"],
            "genre_Aminus": (Y["G"] - Y["none"]) / Y["G"],
            "interaction": (Y["AG"] - Y["A"]) / Y["AG"] - (Y["G"] - Y["none"]) / Y["G"]}


def bca(theta_hat, boot, jack):
    b = boot[np.isfinite(boot)]
    z0 = norm.ppf(np.clip((b < theta_hat).mean() + 0.5 * (b == theta_hat).mean(), 1e-6, 1 - 1e-6))
    jm = jack.mean()
    num, den = ((jm - jack) ** 3).sum(), 6 * (((jm - jack) ** 2).sum()) ** 1.5
    acc = num / den if den > 0 else 0.0
    out = []
    for q in (0.025, 0.975):
        zq = norm.ppf(q)
        out.append(np.quantile(b, norm.cdf(z0 + (z0 + zq) / (1 - acc * (z0 + zq)))))
    return out, z0, acc


rows, perstyle, scale_rows = [], [], []
dsets = a.datasets.split(",")
for ds in dsets:
    for sub in ["preds", "preds_lr"]:
        pdir = ARTIFACTS / "results" / sub / ds
        if not pdir.exists():
            continue
        for f in sorted(pdir.glob("*.parquet")):
            enc = f.stem
            P = pd.read_parquet(f)
            if sub == "preds":   # logistic regression comes only from preds_lr (one consistent run, C7)
                P = P[P.evaluator != "logreg"]
            if EXCL is not None:
                P = P[[(ds, s, i) not in EXCL for s, i in zip(P.seed, P.image_id)]]
            for design, cmap in DESIGNS.items():
                for ev in sorted(P.evaluator.unique()):
                    Q = P[(P.evaluator == ev) & P.protocol.isin(cmap.values())]
                    if Q.protocol.nunique() < 4:
                        continue
                    seeds = sorted(Q.seed.unique())
                    point = {c: [] for c in cmap}
                    boot = {c: np.zeros((R, len(seeds))) for c in cmap}
                    boot_nan = {c: np.zeros((R, len(seeds))) for c in cmap}
                    dropped = np.zeros((R, len(seeds)), bool)
                    jack_seed = []          # per seed: dict cond -> array over artists (seed effect without artist j)
                    ps_rec = []
                    for si, s in enumerate(seeds):
                        qs = Q[Q.seed == s]
                        arts = sorted(qs.artist.unique())
                        stys = sorted(qs["style"].unique())
                        ai = {x: i for i, x in enumerate(arts)}
                        st = {x: i for i, x in enumerate(stys)}
                        base = qs[qs.protocol == cmap["AG"]]
                        N = np.zeros((len(arts), len(stys)))
                        np.add.at(N, (base.artist.map(ai).to_numpy(), base["style"].map(st).to_numpy()), 1)
                        W = rng.multinomial(len(arts), np.ones(len(arts)) / len(arts), size=R).astype(float)
                        WN = W @ N
                        dropped[:, si] = (WN == 0).any(1)
                        js = {}
                        for c, proto in cmap.items():
                            d = qs[qs.protocol == proto]
                            C = np.zeros_like(N)
                            np.add.at(C, (d.artist.map(ai).to_numpy(), d["style"].map(st).to_numpy()),
                                      (d.y == d.pred).to_numpy().astype(float))
                            rec = C.sum(0) / N.sum(0)
                            point[c].append(rec.mean())
                            with np.errstate(invalid="ignore", divide="ignore"):
                                rb = (W @ C) / WN
                                Cl = C.sum(0)[None, :] - C
                                Nl = N.sum(0)[None, :] - N
                                rj = np.where(Nl > 0, Cl / np.where(Nl > 0, Nl, 1), np.nan)
                            # a style that lost all its test images in a replicate keeps its point recall, so the
                            # set of styles averaged in balanced accuracy is constant across replicates
                            boot[c][:, si] = np.where(WN > 0, rb, rec[None, :]).mean(1)
                            boot_nan[c][:, si] = np.nanmean(rb, axis=1)   # alternative: average remaining styles
                            js[c] = np.nanmean(rj, axis=1)   # leave-artist-j-out balanced accuracy of seed s
                            ps_rec.append(pd.DataFrame({"style": stys, "cond": c, "seed": s, "recall": rec}))
                        jack_seed.append(js)
                    Yp = {c: np.array(v) for c, v in point.items()}
                    pt = effects(Yp)
                    sb_idx = rng.integers(len(seeds), size=(R, len(seeds)))
                    hb = {c: np.take_along_axis(boot[c], sb_idx, 1).mean(1) for c in cmap}
                    nodrop = ~np.take_along_axis(dropped, sb_idx, 1).any(1)
                    hbe = effects(hb)
                    hbe_nan = effects({c: np.take_along_axis(boot_nan[c], sb_idx, 1).mean(1) for c in cmap})
                    # jackknife over (seed, artist) clusters of the overall effect (mean over seeds)
                    seed_eff = {k: v for k, v in pt.items()}
                    jack = {k: [] for k in pt}
                    for si, js in enumerate(jack_seed):
                        e_minus = effects({c: js[c] for c in cmap})
                        for k in pt:
                            tot = seed_eff[k].sum() - seed_eff[k][si]
                            jack[k].append((tot + e_minus[k]) / len(seeds))
                    for k, v in pt.items():
                        sbv = v[sb_idx].mean(1)
                        (blo, bhi), z0, acc = bca(v.mean(), hbe[k], np.concatenate(jack[k]))
                        rows.append({"dataset": ds, "encoder": enc, "design": design, "evaluator": ev, "effect": k,
                                     "mean": v.mean(), "seed_lo": np.quantile(sbv, .025), "seed_hi": np.quantile(sbv, .975),
                                     "hier_lo": np.quantile(hbe[k], .025), "hier_hi": np.quantile(hbe[k], .975),
                                     "bca_lo": blo, "bca_hi": bhi, "boot_bias": hbe[k].mean() - v.mean(),
                                     "bca_z0": z0, "bca_acc": acc, "share_reps_style_dropped": 1 - nodrop.mean(),
                                     "share_style_drops_per_seed_rep": float(dropped.mean()),
                                     "alt_nan_lo": np.quantile(hbe_nan[k], .025), "alt_nan_hi": np.quantile(hbe_nan[k], .975),
                                     "n_seeds": len(seeds)})
                    lg = effects({c: logit(v) for c, v in Yp.items()})
                    rl = rel(Yp)
                    for k in pt:
                        scale_rows.append({"dataset": ds, "encoder": enc, "design": design, "evaluator": ev,
                                           "effect": k, "bacc_pts": 100 * pt[k].mean(), "logit": lg[k].mean(),
                                           "relative_pct": 100 * rl[k].mean(),
                                           "logit_seed_pos": int((lg[k] > 0).sum()), "n_seeds": len(seeds)})
                    psd = pd.concat(ps_rec).pivot_table(index=["style", "seed"], columns="cond", values="recall")
                    for k, v in effects({c: psd[c] for c in cmap}).items():
                        perstyle.append(pd.DataFrame({"dataset": ds, "encoder": enc, "design": design, "evaluator": ev,
                                                      "effect": k, "style": v.index.get_level_values(0), "value": v.values}))
res = pd.DataFrame(rows)
res.to_csv(OUT / "effects.csv", index=False)
pd.DataFrame(scale_rows).to_csv(OUT / "effects_scales.csv", index=False)
pd.concat(perstyle).to_csv(OUT / "perstyle_effects.csv", index=False)
# exposure statistics per condition (from split files)
expo = []
for ds in dsets:
    for design, cmap in DESIGNS.items():
        for c, proto in cmap.items():
            for f in sorted((ARTIFACTS / "splits" / ds / proto).glob("seed*.parquet")) if ds != "mp_novel" else []:
                sp = pd.read_parquet(f)
                te, tr = sp[sp.part == "test"], sp[sp.part != "test"]
                ta = set(te["artist"])
                per = tr[tr["artist"].isin(ta)].groupby("artist").size()
                expo.append({"dataset": ds, "design": design, "cond": c, "seed": int(f.stem[4:]),
                             "test_artists": len(ta), "test_artists_seen": int(per.size),
                             "train_imgs_per_seen_artist_median": float(per.median()) if per.size else 0.0,
                             "train_imgs_of_test_artists": int(per.sum()),
                             "test_genre_imgs_in_train": int(tr["genre"].isin(set(te["genre"])).sum())})
pd.DataFrame(expo).to_csv(OUT / "exposure.csv", index=False)
summ = {}
for (ds, design, ev), g in res.groupby(["dataset", "design", "evaluator"]):
    gm = g.set_index(["encoder", "effect"])
    am, it, ap_ = (gm.xs(k, level=1) for k in ["genre_Aminus", "interaction", "genre_Aplus"])
    summ[f"{ds}|{design}|{ev}"] = {
        "n_encoders": int(len(am)),
        "genre_Aplus_mean": float(ap_["mean"].mean()), "genre_Aminus_mean": float(am["mean"].mean()),
        "artist_mean": float(gm.xs("artist", level=1)["mean"].mean()), "interaction_mean": float(it["mean"].mean()),
        "interaction_pos_seed": int((it.seed_lo > 0).sum()), "interaction_pos_hier": int((it.hier_lo > 0).sum()),
        "interaction_pos_bca": int((it.bca_lo > 0).sum()),
        "genre_Aminus_pos_hier": int((am.hier_lo > 0).sum()), "genre_Aminus_pos_bca": int((am.bca_lo > 0).sum()),
        "genre_Aminus_upper_below_delta_hier": int((am.hier_hi < DELTA).sum()),
        "genre_Aminus_upper_below_delta_bca": int((am.bca_hi < DELTA).sum()),
        "genre_Aminus_max_upper_bca": float(am.bca_hi.max()),
        "ratio_Aminus_over_Aplus": float(am["mean"].mean() / ap_["mean"].mean()),
        "interaction_bias_mean": float(it.boot_bias.mean())}
json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if "knn5" in k}, indent=1))
