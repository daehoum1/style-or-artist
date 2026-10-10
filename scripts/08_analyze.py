"""Tables, figures, correlations and the master results file for one dataset.
Usage: 08_analyze.py <dataset> [--split_root X]   -> artifacts/final/<split_root>/"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.analysis.correlations import correlation_table  # noqa: E402
from stylebench.config import ARTIFACTS, load_yaml  # noqa: E402
from stylebench.evaluation.statistics import boot_ci, kendall_tau, paired_summary  # noqa: E402
from stylebench.models.base import encoder_configs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--split_root", default=None)
a = ap.parse_args()
sr = a.split_root or a.dataset
A = load_yaml("configs/experiments/analysis.yaml")
ENC = encoder_configs()
OUT = ARTIFACTS / "final" / sr
OUT.mkdir(parents=True, exist_ok=True)
NICE = {"clip_b16": "CLIP B/16", "clip_l14": "CLIP L/14", "openclip_b16": "OpenCLIP B/16 (LAION-2B)",
        "siglip_b16": "SigLIP B/16", "siglip_so400m": "SigLIP SO400M", "siglip2_l16": "SigLIP2 L/16",
        "dinov2_b14": "DINOv2 B/14", "dinov2_l14": "DINOv2 L/14", "dinov3_b16": "DINOv3 B/16",
        "mae_b16": "MAE B/16", "vit_b16_in21k": "ViT-B/16 (IN-21k sup.)", "csd_l14": "CSD L/14",
        "aladin_vit": "ALADIN-ViT"}
PROTO_NICE = {"iid": "IID", "artist_disjoint": "Artist-D", "genre_disjoint": "Genre-D", "joint": "Artist+Genre-D",
              "mf_iid": "MF: both seen", "mf_artist": "MF: artist unseen", "mf_genre": "MF: genre unseen",
              "mf_joint": "MF: both unseen"}

se = pd.concat([pd.read_parquet(f) for f in sorted((ARTIFACTS / "results" / "style_eval" / sr).glob("*.parquet"))
                if not f.stem.endswith("_per_class")], ignore_index=True)
encs = [e for e in ENC if e in set(se.encoder)]
probes_dir = ARTIFACTS / "results" / "probes" / a.dataset
pr = pd.concat([pd.read_parquet(f) for f in sorted(probes_dir.glob("*.parquet"))], ignore_index=True) \
    if probes_dir.exists() and list(probes_dir.glob("*.parquet")) else pd.DataFrame()
protos = [p for p in PROTO_NICE if p in set(se.protocol)]
has_genre = "genre_disjoint" in protos
METRICS = A["metrics"]


def fmt(m, s):
    return f"{100*m:.1f} ± {100*s:.1f}"


# ------------------------------------------------------------------ Table 2 (+ gaps), per evaluator/metric
gap_rows, main_rows = [], []
for ev in [e for e in [A["primary_evaluator"]] + A["secondary_evaluators"] if e in set(se.evaluator)]:
    for met in METRICS:
        for e in encs:
            d = se[(se.encoder == e) & (se.evaluator == ev)]
            piv = d.pivot_table(index="seed", columns="protocol", values=met)
            row = {"evaluator": ev, "metric": met, "encoder": e, "family": ENC[e]["family"]}
            for p in protos:
                if p in piv:
                    row[p + "_mean"], row[p + "_sd"] = piv[p].mean(), piv[p].std(ddof=1)
                    row[p + "_ci_lo"], row[p + "_ci_hi"] = boot_ci(piv[p].values)
            for g, (x, y) in {**A["gaps"], **A["mf_effects"]}.items():
                if x in piv and y in piv:
                    ps = paired_summary(piv[x].values, piv[y].values)
                    row[g] = ps["mean"]
                    row[g + "_rel"] = (piv[x] - piv[y]).mean() / piv[x].mean()
                    gap_rows.append({"evaluator": ev, "metric": met, "encoder": e, "gap": g, **ps})
            main_rows.append(row)
main = pd.DataFrame(main_rows)
gaps = pd.DataFrame(gap_rows)
main.to_csv(OUT / "table2_main_all.csv", index=False)
gaps.to_csv(OUT / "gaps_paired_all.csv", index=False)


def latex_main(ev, met, cols, gcols, fname, caption):
    d = main[(main.evaluator == ev) & (main.metric == met)].set_index("encoder").loc[encs]
    lines = [r"\begin{tabular}{l" + "c" * (len(cols) + len(gcols)) + "}", r"\toprule",
             "Encoder & " + " & ".join(PROTO_NICE[c] for c in cols) + " & " +
             " & ".join(g.replace("_", " ").title() for g in gcols) + r" \\", r"\midrule"]
    csv = []
    for e, r in d.iterrows():
        cells = [fmt(r[c + "_mean"], r[c + "_sd"]) for c in cols]
        gcell = []
        for g in gcols:
            gs = gaps[(gaps.evaluator == ev) & (gaps.metric == met) & (gaps.encoder == e) & (gaps.gap == g)].iloc[0]
            gcell.append(f"{100*gs['mean']:.1f} [{100*gs.ci_lo:.1f}, {100*gs.ci_hi:.1f}]")
        lines.append(NICE[e] + " & " + " & ".join(cells + gcell) + r" \\")
        csv.append({"encoder": NICE[e], **dict(zip([PROTO_NICE[c] for c in cols], cells)), **dict(zip(gcols, gcell))})
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"{fname}.tex").write_text(f"% {caption}\n" + "\n".join(lines) + "\n")
    pd.DataFrame(csv).to_csv(OUT / f"{fname}.csv", index=False)


std_cols = [p for p in ["iid", "artist_disjoint", "genre_disjoint", "joint"] if p in protos]
std_gaps = [g for g in ["artist_gap", "genre_gap", "joint_gap"] if A["gaps"][g][1] in protos]
mf_cols = [p for p in ["mf_iid", "mf_artist", "mf_genre", "mf_joint"] if p in protos]
for ev in [e for e in [A["primary_evaluator"], "logreg"] if e in set(se.evaluator)]:
    for met in METRICS:
        latex_main(ev, met, std_cols, std_gaps, f"table2_{ev}_{met}",
                   f"{sr}: {ev}, {met} (%), mean ± sd over seeds; gaps: paired mean [95% bootstrap CI]")
        if mf_cols:
            latex_main(ev, met, mf_cols, ["artist_effect", "genre_effect", "joint_effect"],
                       f"table2mf_{ev}_{met}", f"{sr}: matched factorial, {ev}, {met} (%)")

# ------------------------------------------------------------------ Table 3: probes
enc_level = main[(main.evaluator == A["primary_evaluator"]) & (main.metric == A["primary_metric"])].set_index("encoder")
probe_tab = pd.DataFrame(index=encs)
if len(pr):
    for ev in ["logreg", "knn5"]:
        q = pr[pr.evaluator == ev]
        for pname in ["style", "artist", "genre", "within_style_artist", "within_style_genre"]:
            sub = q[q.probe == pname]
            if not len(sub):
                continue
            for met in ["balanced_accuracy_norm", "balanced_accuracy", "macro_f1"]:
                per_seed = sub.groupby(["encoder", "seed"])[met].mean()   # within-style: mean over styles
                probe_tab[f"{ev}:{pname}:{met}"] = per_seed.groupby("encoder").mean()
                probe_tab[f"{ev}:{pname}:{met}:sd"] = per_seed.groupby("encoder").std(ddof=1)
        if (q.probe == "within_style_artist").any():
            probe_tab[f"{ev}:n_styles_within_artist"] = q[q.probe == "within_style_artist"].groupby("encoder")["style"].nunique()
        if (q.probe == "within_style_genre").any():
            probe_tab[f"{ev}:n_styles_within_genre"] = q[q.probe == "within_style_genre"].groupby("encoder")["style"].nunique()
    probe_tab.to_csv(OUT / "table3_probes_all.csv")
    pe, pm = A["probe_evaluator"], A["probe_metric"]
    cols3 = [c for c in ["style", "artist", "within_style_artist", "genre", "within_style_genre"]
             if f"{pe}:{c}:{pm}" in probe_tab]
    lines = [r"\begin{tabular}{l" + "c" * len(cols3) + "}", r"\toprule",
             "Encoder & " + " & ".join(c.replace("_", " ").title() for c in cols3) + r" \\", r"\midrule"]
    for e in encs:
        lines.append(NICE[e] + " & " + " & ".join(
            f"{100*probe_tab.loc[e, f'{pe}:{c}:{pm}']:.1f} ± {100*probe_tab.loc[e, f'{pe}:{c}:{pm}:sd']:.1f}" for c in cols3) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table3_probes.tex").write_text(f"% {sr}: probe {pe}, chance-normalized balanced accuracy (%), mean ± sd over probe seeds\n" + "\n".join(lines) + "\n")

# ------------------------------------------------------------------ encoder-level frame + correlations
pe, pm = A["probe_evaluator"], A["probe_metric"]
E = pd.DataFrame(index=encs)
E["family"] = [ENC[e]["family"] for e in encs]
for p in protos:
    E[p] = enc_level[p + "_mean"]
for g in list(A["gaps"]) + list(A["mf_effects"]):
    if g in enc_level:
        E[g] = enc_level[g]
        E[g + "_rel"] = enc_level[g + "_rel"]
for c in ["style", "artist", "genre", "within_style_artist", "within_style_genre"]:
    if f"{pe}:{c}:{pm}" in probe_tab:
        E[c] = probe_tab[f"{pe}:{c}:{pm}"]
if "within_style_artist" in E and "within_style_genre" in E:
    E["combined_within_style"] = (E["within_style_artist"] + E["within_style_genre"]) / 2
if "artist" in E and "genre" in E:
    E["combined_global"] = (E["artist"] + E["genre"]) / 2
E.to_csv(OUT / "encoder_level.csv")
corr = {}
if len(E) >= 4 and "within_style_artist" in E:
    prim = [t for t in A["primary_tests"] if t["leak"] in E and t["gap"] in E]
    corr["primary"] = correlation_table(E, prim, control="iid")
    sec = []
    for leak in ["artist", "within_style_artist", "genre", "within_style_genre", "combined_within_style", "combined_global"]:
        for gap in ["artist_gap", "genre_gap", "joint_gap", "artist_gap_rel", "genre_gap_rel", "joint_gap_rel",
                    "artist_effect", "genre_effect", "joint_effect", "artist_effect_rel", "genre_effect_rel", "joint_effect_rel"]:
            if leak in E and gap in E:
                sec.append({"leak": leak, "gap": gap})
    corr["secondary"] = correlation_table(E, sec, control="iid")
    for k, v in corr.items():
        v.to_csv(OUT / f"correlations_{k}.csv", index=False)

# ------------------------------------------------------------------ rankings
rank = {}
if "joint" in E:
    rank["kendall_tau_iid_vs_joint"] = kendall_tau(E["iid"], E["joint"])
    rank["kendall_tau_iid_vs_artist"] = kendall_tau(E["iid"], E["artist_disjoint"])
    if "genre_disjoint" in E:
        rank["kendall_tau_iid_vs_genre"] = kendall_tau(E["iid"], E["genre_disjoint"])
    # seed-level stability: tau per seed
    d = se[(se.evaluator == A["primary_evaluator"])]
    taus = []
    for s in sorted(d.seed.unique()):
        x = d[(d.seed == s) & (d.protocol == "iid")].set_index("encoder")[A["primary_metric"]]
        y = d[(d.seed == s) & (d.protocol == "joint")].set_index("encoder")[A["primary_metric"]]
        j = x.index.intersection(y.index)
        if len(j) > 3:
            taus.append(kendall_tau(x[j], y[j]))
    rank["kendall_tau_iid_vs_joint_per_seed"] = taus
    rank["rank_iid"] = E["iid"].rank(ascending=False).astype(int).to_dict()
    rank["rank_joint"] = E["joint"].rank(ascending=False).astype(int).to_dict()
elif "artist_disjoint" in E:
    rank["kendall_tau_iid_vs_artist"] = kendall_tau(E["iid"], E["artist_disjoint"])
    rank["rank_iid"] = E["iid"].rank(ascending=False).astype(int).to_dict()
    rank["rank_artist"] = E["artist_disjoint"].rank(ascending=False).astype(int).to_dict()

# ------------------------------------------------------------------ figures
FAM_COL = {"vl": "#1f77b4", "ssl": "#2ca02c", "sup": "#9467bd", "art": "#d62728"}
FAM_NAME = {"vl": "vision-language", "ssl": "self-supervised", "sup": "supervised", "art": "art-specific"}
plt.rcParams.update({"font.size": 9})
ev, met = A["primary_evaluator"], A["primary_metric"]
for which, cols, fname in [("standard", std_cols, "fig1_controlled_evaluation"), ("mf", mf_cols, "fig1b_matched_factorial")]:
    if not cols:
        continue
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for e in encs:
        r = main[(main.evaluator == ev) & (main.metric == met) & (main.encoder == e)].iloc[0]
        ax.errorbar(range(len(cols)), [100 * r[c + "_mean"] for c in cols], yerr=[100 * r[c + "_sd"] for c in cols],
                    marker="o", ms=3, lw=1, capsize=2, color=FAM_COL[ENC[e]["family"]], alpha=0.85)
        ax.annotate(NICE[e], (len(cols) - 1, 100 * r[cols[-1] + "_mean"]), fontsize=6, xytext=(4, 0),
                    textcoords="offset points", va="center")
    ax.set_xticks(range(len(cols)), [PROTO_NICE[c] for c in cols], rotation=15)
    ax.set_ylabel(f"{met.replace('_', ' ')} (%) — {ev}")
    ax.set_xlim(-0.3, len(cols) + 0.9)
    for f, c in FAM_COL.items():
        ax.plot([], [], color=c, label=FAM_NAME[f])
    ax.legend(fontsize=6, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png", dpi=200)
    plt.close(fig)


def scatter(xcol, ycol, fname, xl, yl):
    if xcol not in E or ycol not in E:
        return
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for e in encs:
        ax.scatter(100 * E.loc[e, xcol], 100 * E.loc[e, ycol], color=FAM_COL[ENC[e]["family"]], s=18)
        ax.annotate(NICE[e], (100 * E.loc[e, xcol], 100 * E.loc[e, ycol]), fontsize=5.5, xytext=(3, 2),
                    textcoords="offset points")
    allc = pd.concat([v for v in corr.values()]) if corr else pd.DataFrame()
    t = allc[(allc.leak == xcol) & (allc.gap == ycol)] if len(allc) else allc
    if len(t):
        t = t.iloc[0]
        ax.set_title(f"Spearman ρ={t.rho:.2f} [{t.ci_lo:.2f}, {t.ci_hi:.2f}], perm p={t.perm_p:.3f}", fontsize=7)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png", dpi=200)
    plt.close(fig)


scatter("within_style_artist", "artist_gap", "fig2_artist_leak_vs_gap", "within-style artist probe (norm. bAcc, %)", "Artist gap (bAcc pts)")
scatter("artist", "artist_gap", "fig2b_global_artist_leak_vs_gap", "global artist probe (norm. bAcc, %)", "Artist gap (bAcc pts)")
scatter("within_style_genre", "genre_gap", "fig3_genre_leak_vs_gap", "within-style genre probe (norm. bAcc, %)", "Genre gap (bAcc pts)")
scatter("genre", "genre_gap", "fig3b_global_genre_leak_vs_gap", "global genre probe (norm. bAcc, %)", "Genre gap (bAcc pts)")
scatter("combined_within_style", "joint_gap", "fig4_combined_leak_vs_joint_gap", "combined within-style leakage (%)", "Joint gap (bAcc pts)")
scatter("within_style_artist", "artist_effect", "fig2c_artist_leak_vs_mf_effect", "within-style artist probe (%)", "MF artist effect (bAcc pts)")
scatter("within_style_genre", "genre_effect", "fig3c_genre_leak_vs_mf_effect", "within-style genre probe (%)", "MF genre effect (bAcc pts)")
last = "joint" if "joint" in E else "artist_disjoint"
fig, ax = plt.subplots(figsize=(3.8, 3.8))
ri, rj = E["iid"].rank(ascending=False), E[last].rank(ascending=False)
for e in encs:
    ax.plot([0, 1], [ri[e], rj[e]], marker="o", color=FAM_COL[ENC[e]["family"]])
    ax.annotate(NICE[e], (0, ri[e]), xytext=(-4, 0), textcoords="offset points", ha="right", va="center", fontsize=6)
    ax.annotate(NICE[e], (1, rj[e]), xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=6)
ax.invert_yaxis()
ax.set_xticks([0, 1], ["IID", PROTO_NICE[last]])
ax.set_xlim(-0.9, 1.9)
ax.set_yticks([])
ax.set_title(f"rank change, Kendall τ={kendall_tau(E['iid'], E[last]):.2f}", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig5_ranking.pdf")
fig.savefig(OUT / "fig5_ranking.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ master results
master = {"dataset": a.dataset, "split_root": sr, "encoders": encs, "protocols": protos,
          "analysis_config": A, "encoder_level": json.loads(E.to_json(orient="index")),
          "correlations": {k: json.loads(v.to_json(orient="records")) for k, v in corr.items()},
          "rankings": rank}
json.dump(master, open(OUT / "master_results.json", "w"), indent=1, default=float)
se.to_csv(OUT / "all_style_runs.csv", index=False)
if len(pr):
    pr.to_csv(OUT / "all_probe_runs.csv", index=False)
print(E.round(3).to_string())
for k, v in corr.items():
    print(f"\n{k} correlations\n", v.round(3).to_string())
print(json.dumps({k: v for k, v in rank.items() if "per_seed" not in k}, indent=0, default=str))
