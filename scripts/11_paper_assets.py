"""Every number, table and figure of the manuscript: paper/numbers.tex, paper/tables/*.tex, paper/figures/*.pdf.
Reads only artifacts/ (no manual numbers in the manuscript)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stylebench.evaluation.statistics import paired_summary  # noqa: E402

A = ROOT / "artifacts"
F = A / "final"
P = ROOT / "paper"
(P / "tables").mkdir(parents=True, exist_ok=True)
(P / "figures").mkdir(parents=True, exist_ok=True)
DS = {"wikiart": "WA", "multitask_painting": "MP"}
DSN = {"wikiart": "WikiArt", "multitask_painting": "MultitaskPainting100k"}
ENC = ["clip_b16", "clip_l14", "openclip_b16", "siglip_b16", "siglip2_l16", "siglip_so400m",
       "dinov2_b14", "dinov2_l14", "dinov3_b16", "mae_b16", "vit_b16_in21k", "csd_l14", "aladin_vit"]
NICE = {"clip_b16": "CLIP B/16", "clip_l14": "CLIP L/14", "openclip_b16": "OpenCLIP B/16",
        "siglip_b16": "SigLIP B/16", "siglip2_l16": "SigLIP2 L/16", "siglip_so400m": "SigLIP SO400M",
        "dinov2_b14": "DINOv2 B/14", "dinov2_l14": "DINOv2 L/14", "dinov3_b16": "DINOv3 B/16",
        "mae_b16": "MAE B/16", "vit_b16_in21k": "ViT-B/16 (sup.)", "csd_l14": "CSD L/14", "aladin_vit": "ALADIN-ViT"}
FAM = {"clip_b16": "vl", "clip_l14": "vl", "openclip_b16": "vl", "siglip_b16": "vl", "siglip2_l16": "vl",
       "siglip_so400m": "vl", "dinov2_b14": "ssl", "dinov2_l14": "ssl", "dinov3_b16": "ssl", "mae_b16": "ssl",
       "vit_b16_in21k": "sup", "csd_l14": "art", "aladin_vit": "art"}
# validated categorical palette (dataviz reference palette slots 1-4; CVD/normal-vision checks pass)
FAMCOL = {"vl": "#2a78d6", "ssl": "#eb6834", "sup": "#1baf7a", "art": "#eda100"}
FAMNAME = {"vl": "vision-language", "ssl": "self-supervised", "sup": "supervised", "art": "art-specific"}
FAMMARK = {"vl": "o", "ssl": "s", "sup": "^", "art": "D"}


def f1(x, d=1):
    """Fixed-point formatting without a negative zero (A10)."""
    v = round(float(x), d)
    return f"{0.0:.{d}f}" if v == 0 else f"{v:.{d}f}"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42})

M = {}  # macro name -> string


def num(x, d=1):
    return f"{x:.{d}f}"


def mac(name, val):
    assert name.isalpha(), name
    M[name] = val


# ------------------------------------------------------------------ load
E, runs, gaps, probes = {}, {}, {}, {}
for d in DS:
    E[d] = pd.read_csv(F / d / "encoder_level.csv", index_col=0).loc[ENC]
    runs[d] = pd.read_csv(F / d / "all_style_runs.csv")
    gaps[d] = pd.read_csv(F / d / "gaps_paired_all.csv")
    probes[d] = pd.read_csv(F / d / "all_probe_runs.csv")

# ------------------------------------------------------------------ dataset / cleaning numbers
for d, t in DS.items():
    au = json.load(open(A / "reports" / f"{d}_audit.json"))
    cs = au["cleaning"]
    ss = pd.read_csv(A / "splits" / d / "split_summary.csv")
    core = pd.read_csv(A / "splits" / d / "core_style_table.csv", index_col=0)
    cand = json.load(open(A / "splits" / d / "genre_holdout_candidates.json"))
    mac(f"{t}raw", f"{au['raw']['n_images']:,}")
    mac(f"{t}clean", f"{au['cleaned']['n_images']:,}")
    mac(f"{t}rawstyles", str(au["raw"]["n_styles"]))
    mac(f"{t}artists", f"{au['cleaned']['n_artists']:,}")
    mac(f"{t}genres", str(au["cleaned"]["n_genres"]))
    mac(f"{t}missgenre", f"{au['cleaned']['missing']['genre']:,}")
    mac(f"{t}exactdup", f"{cs['n_exact_duplicate_images']:,}")
    mac(f"{t}nearpairs", f"{cs['n_near_duplicate_pairs']:,}")
    rc = cs["reason_counts"]
    mac(f"{t}dropstyleconf", f"{rc.get('exact_duplicate_group_conflicting_style',0)+rc.get('near_duplicate_group_conflicting_style',0):,}")
    mac(f"{t}dropartconf", f"{rc.get('exact_duplicate_group_conflicting_artist',0)+rc.get('near_duplicate_group_conflicting_artist',0):,}")
    mac(f"{t}dropdup", f"{rc.get('exact_duplicate_of',0)+rc.get('near_duplicate_of',0):,}")
    mac(f"{t}pool", f"{int(ss[(ss.protocol=='iid')&(ss.seed==0)][['n_train_total','n_test']].sum(1).iloc[0]):,}")
    mac(f"{t}core", str(int((core.excluded_reason.fillna('') == '').sum())))
    mac(f"{t}cand", str(cand["n_candidates"]))
    for p in ["iid", "artist_disjoint", "genre_disjoint", "joint", "mf_joint"]:
        q = ss[ss.protocol == p]
        key = {"iid": "Iid", "artist_disjoint": "Art", "genre_disjoint": "Gen", "joint": "Joint", "mf_joint": "Mf"}[p]
        mac(f"{t}ntest{key}", f"{int(q.n_test.mean()):,}")
        mac(f"{t}ntrain{key}", f"{int(q.n_train_total.mean()):,}")
    mac(f"{t}jointdiscard", f"{int((ss[(ss.protocol=='iid')].n_train_total.values + ss[ss.protocol=='iid'].n_test.values - ss[ss.protocol=='joint'].n_train_total.values - ss[ss.protocol=='joint'].n_test.values).mean()):,}")

# ------------------------------------------------------------------ main results (kNN-5 bAcc)
PRO = {"iid": "Iid", "artist_disjoint": "Art", "genre_disjoint": "Gen", "joint": "Joint",
       "mf_iid": "MfIid", "mf_artist": "MfArt", "mf_genre": "MfGen", "mf_joint": "MfJoint"}
for d, t in DS.items():
    e = E[d]
    for p, k in PRO.items():
        mac(f"{t}mean{k}", num(100 * e[p].mean()))
        mac(f"{t}best{k}", num(100 * e.loc["siglip_so400m", p]))
        mac(f"{t}min{k}", num(100 * e[p].min()))
    r = e["joint"] / e["iid"]
    mac(f"{t}retlo", num(r.min(), 2)); mac(f"{t}rethi", num(r.max(), 2))
    nm = e.drop("mae_b16")
    mac(f"{t}artefflo", num(100 * nm.artist_effect.min())); mac(f"{t}arteffhi", num(100 * nm.artist_effect.max()))
    mac(f"{t}geneffseenlo", num(100 * nm.genre_effect.min())); mac(f"{t}geneffseenhi", num(100 * nm.genre_effect.max()))
    mac(f"{t}maeartgap", num(100 * e.loc["mae_b16", "artist_effect"]))
    mac(f"{t}artgaplo", num(100 * e.artist_gap.min())); mac(f"{t}artgaphi", num(100 * e.artist_gap.max()))
    mac(f"{t}jointgapmean", num(100 * e.joint_gap.mean()))
    mac(f"{t}csdartgap", num(100 * e.loc["csd_l14", "artist_gap"]))
    mr = json.load(open(F / d / "master_results.json"))["rankings"]
    mac(f"{t}tauJoint", num(mr["kendall_tau_iid_vs_joint"], 2))
    mac(f"{t}tauArt", num(mr["kendall_tau_iid_vs_artist"], 2))
    mac(f"{t}tauGen", num(mr["kendall_tau_iid_vs_genre"], 2))
    ts = mr["kendall_tau_iid_vs_joint_per_seed"]
    mac(f"{t}tauseedlo", num(min(ts), 2)); mac(f"{t}tauseedhi", num(max(ts), 2))
    # logistic regression / other evaluators / metrics, mean over encoders
    rr = runs[d]
    for ev, ek in [("logreg", "Lr"), ("ncc", "Ncc"), ("knn_sel", "Ks")]:
        g = rr[rr.evaluator == ev].groupby(["encoder", "protocol"]).balanced_accuracy.mean().unstack()
        for p, k in PRO.items():
            mac(f"{t}{ek}{k}", num(100 * g[p].mean()))
    for met, mk in [("macro_f1", "Fone"), ("accuracy", "Acc")]:
        g = rr[rr.evaluator == "knn5"].groupby(["encoder", "protocol"])[met].mean().unstack()
        for p, k in PRO.items():
            mac(f"{t}{mk}{k}", num(100 * g[p].mean()))

# ------------------------------------------------------------------ MF interaction (paired over seeds)
inter = []
for d, t in DS.items():
    rr = runs[d][runs[d].evaluator == "knn5"]
    for e in ENC:
        pv = rr[rr.encoder == e].pivot_table(index="seed", columns="protocol", values="balanced_accuracy")
        a = paired_summary(pv.mf_iid, pv.mf_genre)
        b = paired_summary(pv.mf_artist, pv.mf_joint)
        c = paired_summary(pv.mf_iid - pv.mf_genre, pv.mf_artist - pv.mf_joint)
        ar = paired_summary(pv.mf_iid, pv.mf_artist)
        inter.append({"d": d, "e": e, "gs": 100 * a["mean"], "gu": 100 * b["mean"], "gu_lo": 100 * b["ci_lo"],
                      "gu_hi": 100 * b["ci_hi"], "int": 100 * c["mean"], "int_lo": 100 * c["ci_lo"],
                      "int_hi": 100 * c["ci_hi"], "art": 100 * ar["mean"]})
inter = pd.DataFrame(inter)
for d, t in DS.items():
    q = inter[inter.d == d]
    mac(f"{t}gsmean", num(q.gs.mean())); mac(f"{t}gumean", num(q.gu.mean()))
    mac(f"{t}gulo", num(q.gu.min())); mac(f"{t}guhi", num(q.gu.max()))
    mac(f"{t}intmean", num(q.int.mean()))
    mac(f"{t}intpos", str(int((q.int_lo > 0).sum())))
    mac(f"{t}artmean", num(q.art.mean()))
mac("intposall", str(int((inter.int_lo > 0).sum())))
mac("nint", str(len(inter)))
mac("gulo", num(inter.gu.min())); mac("guhi", num(inter.gu.max()))

# ------------------------------------------------------------------ neighbour diagnostic
nd = {}
for d, t in DS.items():
    n = pd.read_csv(A / "results" / "neighbor_diag" / f"{d}.csv")
    nd[d] = n
    g = n.groupby("encoder").mean(numeric_only=True)
    mac(f"{t}ndshare", num(100 * g.share_top1_same_artist.mean(), 0))
    mac(f"{t}ndsharelo", num(100 * g.share_top1_same_artist.min(), 0))
    mac(f"{t}ndsharehi", num(100 * g.share_top1_same_artist.max(), 0))
    mac(f"{t}ndaccsame", num(100 * g.acc_if_top1_same_artist.mean(), 0))
    mac(f"{t}ndaccother", num(100 * g.acc_if_top1_other_artist.mean(), 0))
    mac(f"{t}ndaccnone", num(100 * g.acc_if_no_same_artist_in_5.mean(), 0))
    mac(f"{t}ndacc", num(100 * g.acc.mean(), 0))
    mac(f"{t}ndsamegenreother", num(100 * g.share_top1_same_genre_other_artist.mean(), 0))
    mac(f"{t}ndaccsg", num(100 * g.acc_if_top1_same_genre.mean(), 0))
    mac(f"{t}ndaccog", num(100 * g.acc_if_top1_other_genre.mean(), 0))
    a = runs[d][(runs[d].evaluator == "knn5") & (runs[d].protocol == "artist_disjoint")].accuracy.mean()
    mac(f"{t}artaccmean", num(100 * a, 0))
    from scipy.stats import spearmanr as _sp
    mac(f"{t}rhoShareIid", num(_sp(E[d].loc[g.index, "iid"], g.share_top1_same_artist).statistic, 2))
    exc = [NICE[k] for k in ENC if E[d].loc[k, "genre_disjoint"] < E[d].loc[k, "artist_disjoint"]]
    mac(f"{t}nGenBelowArt", str(len(exc)))

# ------------------------------------------------------------------ probes + correlations
PROBES = ["style", "artist", "within_style_artist", "genre", "within_style_genre"]
for d, t in DS.items():
    e = E[d]
    for p, k in zip(PROBES, ["Sty", "Art", "WArt", "Gen", "WGen"]):
        mac(f"{t}probe{k}lo", num(100 * e[p].min())); mac(f"{t}probe{k}hi", num(100 * e[p].max()))
    q = probes[d]
    mac(f"{t}nstylesWArt", str(q[q.probe == "within_style_artist"]["style"].nunique()))
    mac(f"{t}nstylesWGen", str(q[q.probe == "within_style_genre"]["style"].nunique()))
    mac(f"{t}nartistclasses", str(int(q[(q.probe == "artist")].n_classes.iloc[0])))
    c = pd.read_csv(F / d / "correlations_primary.csv").set_index("gap")
    for g, k in [("artist_gap", "A"), ("genre_gap", "G"), ("joint_gap", "J")]:
        r = c.loc[g]
        mac(f"{t}rho{k}", num(r.rho, 2)); mac(f"{t}rho{k}lo", num(r.ci_lo, 2)); mac(f"{t}rho{k}hi", num(r.ci_hi, 2))
        mac(f"{t}p{k}", num(r.perm_p_holm, 3))
        mac(f"{t}prho{k}", num(r["partial_rho|iid"], 2)); mac(f"{t}prho{k}lo", num(r.partial_ci_lo, 2))
        mac(f"{t}prho{k}hi", num(r.partial_ci_hi, 2))
    sec = pd.read_csv(F / d / "correlations_secondary.csv")
    s = sec[(sec.leak == "within_style_artist") & (sec.gap == "artist_gap_rel")].iloc[0]
    mac(f"{t}rhoRel", num(s.rho, 2)); mac(f"{t}rhoRello", num(s.ci_lo, 2)); mac(f"{t}rhoRelhi", num(s.ci_hi, 2))
    s = sec[(sec.leak == "within_style_artist") & (sec.gap == "artist_effect")].iloc[0]
    mac(f"{t}rhoMf", num(s.rho, 2)); mac(f"{t}prhoMf", num(s["partial_rho|iid"], 2))
    from scipy.stats import spearmanr
    mac(f"{t}rhoIidArt", num(spearmanr(e["iid"], e["within_style_artist"]).statistic, 2))
    mac(f"{t}rhoIidGap", num(spearmanr(e["iid"], e["artist_gap"]).statistic, 2))

# ------------------------------------------------------------------ sensitivity
for d, t in DS.items():
    for v, vk in [("lenient", "Len"), ("strict", "Str")]:
        ev = pd.read_csv(F / f"{d}_{v}" / "encoder_level.csv", index_col=0)
        for p, k in PRO.items():
            mac(f"{t}{vk}{k}", num(100 * ev[p].mean()))
        core = pd.read_csv(A / "splits" / f"{d}_{v}" / "core_style_table.csv", index_col=0)
        mac(f"{t}{vk}core", str(int((core.excluded_reason.fillna('') == '').sum())))
        mac(f"{t}{vk}tau", num(json.load(open(F / f"{d}_{v}" / "master_results.json"))["rankings"]["kendall_tau_iid_vs_joint"], 2))

# ------------------------------------------------------------------ write macros
with open(P / "numbers.tex", "w") as f:
    f.write("% generated by scripts/11_paper_assets.py -- do not edit\n")
    for k in sorted(M):
        f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
json.dump(M, open(P / "numbers.json", "w"), indent=0)


# ------------------------------------------------------------------ tables
def pm(m, s):
    return f"{100*m:.1f}\\,$\\pm$\\,{100*s:.1f}"


def bold_max(vals):
    mx = max(vals)
    return [f"\\textbf{{{v:.1f}}}" if abs(v - mx) < 1e-9 else f"{v:.1f}" for v in vals]


# Table: main controlled results
lines = [r"\begin{tabular}{l cccc cc}", r"\toprule",
         r"Encoder & IID & Artist-D & Genre-D & Joint & Artist gap & Joint gap \\"]
for d, t in DS.items():
    main = pd.read_csv(F / d / "table2_main_all.csv")
    m = main[(main.evaluator == "knn5") & (main.metric == "balanced_accuracy")].set_index("encoder").loc[ENC]
    lines += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{\textit{{{DSN[d]}}}}} \\"]
    for e in ENC:
        r = m.loc[e]
        cells = [pm(r[p + "_mean"], r[p + "_sd"]) for p in ["iid", "artist_disjoint", "genre_disjoint", "joint"]]
        cells += [num(100 * r.artist_gap), num(100 * r.joint_gap)]
        lines.append(NICE[e] + " & " + " & ".join(cells) + r" \\")
    lines.append(r"Mean & " + " & ".join(num(100 * E[d][p].mean()) for p in ["iid", "artist_disjoint", "genre_disjoint", "joint", "artist_gap", "joint_gap"]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_main.tex").write_text("\n".join(lines) + "\n")

# Table: matched factorial
lines = [r"\begin{tabular}{l cccc ccc}", r"\toprule",
         r" & \multicolumn{4}{c}{Test artists / test genres seen in training} & \multicolumn{3}{c}{Effects (pts)} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
         r"Encoder & both & genres only & artists only & neither & $\Delta_{\mathrm{G}\mid\mathrm{A}^{+}}$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ [95\% CI] & Interaction [95\% CI] \\"]
for d, t in DS.items():
    lines += [r"\midrule", rf"\multicolumn{{8}}{{l}}{{\textit{{{DSN[d]}}}}} \\"]
    for e in ENC:
        r = E[d].loc[e]
        q = inter[(inter.d == d) & (inter.e == e)].iloc[0]
        cells = [num(100 * r[p]) for p in ["mf_iid", "mf_artist", "mf_genre", "mf_joint"]]
        cells += [num(q.gs), f"{q.gu:.1f} [{q.gu_lo:.1f}, {q.gu_hi:.1f}]", f"{q.int:.1f} [{q.int_lo:.1f}, {q.int_hi:.1f}]"]
        lines.append(NICE[e] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_mf.tex").write_text("\n".join(lines) + "\n")

# Table: probes
lines = [r"\begin{tabular}{l ccccc ccccc}", r"\toprule",
         r" & \multicolumn{5}{c}{WikiArt} & \multicolumn{5}{c}{MultitaskPainting100k} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
         r"Encoder & Style & Artist & Artist$\mid$style & Genre & Genre$\mid$style & Style & Artist & Artist$\mid$style & Genre & Genre$\mid$style \\",
         r"\midrule"]
for e in ENC:
    cells = []
    for d in DS:
        cells += [num(100 * E[d].loc[e, p]) for p in PROBES]
    lines.append(NICE[e] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_probes.tex").write_text("\n".join(lines) + "\n")

# Table: correlations
lines = [r"\begin{tabular}{ll cc c}", r"\toprule",
         r"Dataset & Leakage measure $\rightarrow$ gap & $\rho$ [95\% CI] & Holm $p$ & Partial $\rho\mid$IID [95\% CI] \\", r"\midrule"]
names = {"artist_gap": r"Artist$\mid$style $\rightarrow$ Artist gap", "genre_gap": r"Genre$\mid$style $\rightarrow$ Genre gap",
         "joint_gap": r"Combined $\rightarrow$ Joint gap"}
for d in DS:
    c = pd.read_csv(F / d / "correlations_primary.csv")
    for i, r in c.iterrows():
        lines.append((DSN[d] if i == 0 else "") + f" & {names[r.gap]} & {r.rho:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}] & "
                     f"{r.perm_p_holm:.3f} & {r['partial_rho|iid']:.2f} [{r.partial_ci_lo:.2f}, {r.partial_ci_hi:.2f}] \\\\")
    if d == "wikiart":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_corr.tex").write_text("\n".join(lines) + "\n")

# Table: robustness (means over encoders)
lines = [r"\begin{tabular}{ll cccc cccc}", r"\toprule",
         r" & & \multicolumn{4}{c}{Standard protocols} & \multicolumn{4}{c}{Matched factorial (seen: A=artists, G=genres)} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"Dataset & Setting & IID & Artist-D & Genre-D & Joint & A+G & G & A & none \\", r"\midrule"]
rows = [("kNN-5, bAcc (main)", "mean"), ("kNN-5, macro-F1", "Fone"), ("kNN-5, accuracy", "Acc"),
        ("kNN, $k$ on validation", "Ks"), ("Logistic regression", "Lr"), ("Nearest centroid", "Ncc"),
        ("Lenient support, kNN-5", "Len"), ("Strict support, kNN-5", "Str")]
order = ["Iid", "Art", "Gen", "Joint", "MfIid", "MfArt", "MfGen", "MfJoint"]
for d, t in DS.items():
    for i, (lab, k) in enumerate(rows):
        lines.append((DSN[d] if i == 0 else "") + f" & {lab} & " + " & ".join(M[f"{t}{k}{o}"] for o in order) + r" \\")
    if d == "wikiart":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_robust.tex").write_text("\n".join(lines) + "\n")

# Table: encoders
info = {}
for e in ENC:
    c = json.load(open(A / "embeddings" / "wikiart" / e / "config.json"))
    import re as _re
    res = int(_re.findall(r"\d+", str(c.get("input_resolution")))[-1])
    info[e] = (c.get("arch"), c.get("pretrained"), res, c["dim"], c.get("feature"))
src = {"clip_b16": "OpenAI WIT-400M", "clip_l14": "OpenAI WIT-400M", "openclip_b16": "LAION-2B",
       "siglip_b16": "WebLI", "siglip2_l16": "WebLI", "siglip_so400m": "WebLI", "dinov2_b14": "LVD-142M",
       "dinov2_l14": "LVD-142M", "dinov3_b16": "LVD-1689M", "mae_b16": "ImageNet-1k",
       "vit_b16_in21k": "ImageNet-21k (+1k)", "csd_l14": "LAION-Styles (CLIP init.)",
       "aladin_vit": "BAM-FG (weak groups)"}
feat = {"image_embed": "image embedding", "cls_token": "[CLS] token", "cls_token_prelogits": "[CLS] token",
        "mean_patch_tokens": "mean patch token", "style_embed": "style embedding"}
famn = {"vl": "VL", "ssl": "SSL", "sup": "Sup.", "art": "Art"}
lines = [r"\begin{tabular}{l l l l c c l}", r"\toprule",
         r"Encoder & Family & Checkpoint & Pre-training data & Input & Dim. & Feature \\", r"\midrule"]
for e in ENC:
    a, p_, r, dim, ft = info[e]
    ck = {"csd_l14": "tomg-group-umd/CSD-ViT-L", "aladin_vit": "DanRuta/ALADIN v1.0.0"}.get(
        e, a.replace("hf-hub:", "") + (f" ({p_})" if p_ else ""))
    ck = ck.replace("_", r"\_")
    lines.append(f"{NICE[e]} & {famn[FAM[e]]} & \\texttt{{\\scriptsize {ck}}} & {src[e]} & {r} & {dim} & {feat[ft]} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_encoders.tex").write_text("\n".join(lines) + "\n")

# Table: per-style gaps (appendix)
ps = pd.read_csv(F / "per_style_recall_gaps.csv")
from scipy.stats import spearmanr as _sp2
for d, t in DS.items():
    q = ps[ps.dataset == d]
    mac(f"{t}rhoStyleArtists", num(_sp2(q.n_artists, q.artist_gap).statistic, 2))
with open(P / "numbers.tex", "a") as f:
    for d, t in DS.items():
        f.write(f"\\newcommand{{\\{t}rhoStyleArtists}}{{{M[f'{t}rhoStyleArtists']}}}\n")
lines = [r"\begin{tabular}{l l r r cc cc}", r"\toprule",
         r"Dataset & Style & Images & Artists & IID recall & Artist gap & MF artist eff. & MF genre eff. \\", r"\midrule"]
for d in DS:
    q = ps[ps.dataset == d].sort_values("artist_gap", ascending=False)
    for i, (_, r) in enumerate(q.iterrows()):
        lines.append((DSN[d] if i == 0 else "") + f" & {r['style'].title()} & {int(r.n_images):,} & {int(r.n_artists)} & "
                     f"{100*r.iid:.1f} & {100*r.artist_gap:.1f} & "
                     + (f"{100*r.artist_effect:.1f} & {100*r.genre_effect:.1f}" if not np.isnan(r.artist_effect) else "-- & --") + r" \\")
    if d == "wikiart":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_perstyle.tex").write_text("\n".join(lines) + "\n")

# Table: split statistics (appendix)
lines = [r"\begin{tabular}{l l rrr r l}", r"\toprule",
         r"Dataset & Protocol & Train & Val & Test & Test artists & Overlap (artists / genres) \\", r"\midrule"]
pn = {"iid": "IID", "artist_disjoint": "Artist-D", "genre_disjoint": "Genre-D", "joint": "Joint",
      "mf_iid": "MF both seen", "mf_artist": "MF genres seen", "mf_genre": "MF artists seen", "mf_joint": "MF neither"}
for d in DS:
    ss = pd.read_csv(A / "splits" / d / "split_summary.csv")
    for i, p in enumerate(pn):
        q = ss[ss.protocol == p]
        lines.append((DSN[d] if i == 0 else "") + f" & {pn[p]} & {round(q.n_train_fit.mean()):,} & {round(q.n_val.mean()):,} & "
                     f"{round(q.n_test.mean()):,} & {round(q.n_test_artists.mean())} & {q.artist_overlap.mean():.0f} / {q.genre_overlap.mean():.0f} \\\\")
    if d == "wikiart":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_splits.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ figures
def label_ends(ax, xs, ys, labels, colors, x, min_gap):
    """Direct labels at the line ends; overlapping labels are pushed apart symmetrically (no net drift)."""
    ys = np.asarray(ys, float)
    order = np.argsort(ys)
    pos = ys[order].copy()
    for _ in range(500):
        moved = False
        for i in range(1, len(pos)):
            d = pos[i] - pos[i - 1]
            if d < min_gap:
                adj = (min_gap - d) / 2
                pos[i] += adj
                pos[i - 1] -= adj
                moved = True
        if not moved:
            break
    for j, i in enumerate(order):
        ax.annotate(labels[i], (x, ys[i]), xytext=(x + 0.18, pos[j]), textcoords="data", fontsize=6.3,
                    color=INK, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=colors[i], lw=0.6, shrinkA=0, shrinkB=2))


def lines_figure(cols, colnames, fname, ylab):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=False)
    for ax, (d, t) in zip(axes, DS.items()):
        main = pd.read_csv(F / d / "table2_main_all.csv")
        m = main[(main.evaluator == "knn5") & (main.metric == "balanced_accuracy")].set_index("encoder")
        ends = []
        for e in ENC:
            y = [100 * m.loc[e, c + "_mean"] for c in cols]
            s = [100 * m.loc[e, c + "_sd"] for c in cols]
            ax.errorbar(range(len(cols)), y, yerr=s, color=FAMCOL[FAM[e]], lw=1.4, marker=FAMMARK[FAM[e]], ms=3.5,
                        capsize=1.5, elinewidth=0.8, alpha=0.95)
            ends.append(y[-1])
        label_ends(ax, None, ends, [NICE[e] for e in ENC], [FAMCOL[FAM[e]] for e in ENC], len(cols) - 1,
                   (ax.get_ylim()[1] - ax.get_ylim()[0]) / 21)
        ax.set_xticks(range(len(cols)), colnames, fontsize=6.3)
        ax.set_xlim(-0.25, len(cols) + 0.75)
        ax.set_title(DSN[d], fontsize=8.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_ylabel(ylab)
    for f_, c in FAMCOL.items():
        axes[0].plot([], [], color=c, lw=1.6, marker=FAMMARK[f_], ms=4, label=FAMNAME[f_])
    fig.legend(fontsize=6.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(P / "figures" / fname)
    plt.close(fig)


lines_figure(["mf_iid", "mf_genre", "mf_artist", "mf_joint"],
             ["A+G\nseen", "A seen\nG unseen", "G seen\nA unseen", "neither\nseen"],
             "fig_mf.pdf", "balanced accuracy (%), kNN-5")
lines_figure(["iid", "artist_disjoint", "genre_disjoint", "joint"], ["IID", "Artist-D", "Genre-D", "Joint"],
             "fig_protocols.pdf", "balanced accuracy (%), kNN-5")

# neighbour diagnostic: accuracy by neighbour type, one dot per encoder, bar = mean
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
cats = [("acc_if_top1_same_artist", "nearest neighbour\nsame artist"), ("acc_if_top1_other_artist", "nearest neighbour\nother artist"),
        ("acc_if_no_same_artist_in_5", "no same-artist\nimage in top 5")]
for ax, d in zip(axes, DS):
    g = nd[d].groupby("encoder").mean(numeric_only=True).loc[ENC]
    for i, (c, lab) in enumerate(cats):
        ax.bar(i, 100 * g[c].mean(), width=0.6, color="#c3d9f3", edgecolor="none", zorder=1)
        jit = np.linspace(-0.18, 0.18, len(ENC))
        ax.scatter(i + jit, 100 * g[c], s=14, c=[FAMCOL[FAM[e]] for e in ENC], zorder=3, edgecolor="white", lw=0.5)
        ax.text(i, 100 * g[c].max() + 3, f"mean {100*g[c].mean():.0f}", ha="center", fontsize=6.8, color=INK)
    art = runs[d][(runs[d].evaluator == "knn5") & (runs[d].protocol == "artist_disjoint")].groupby("encoder").accuracy.mean().mean()
    ax.axhline(100 * art, color=INK2, lw=0.8, ls="--")
    ax.text(-0.45, 100 * art + 1.5, "mean Artist-D accuracy", fontsize=6, color=INK2, ha="left", va="bottom")
    ax.set_xticks(range(3), [c[1] for c in cats], fontsize=6.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("kNN-5 accuracy, IID (%)")
    sh = 100 * g.share_top1_same_artist.mean()
    ax.set_title(f"{DSN[d]}\n{sh:.0f}% of test images have a same-artist nearest neighbour", fontsize=7.3, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
for f_, c in FAMCOL.items():
    axes[1].scatter([], [], c=c, s=14, label=FAMNAME[f_]) if False else None
    fig.axes[0].scatter([], [], c=c, s=14, label=FAMNAME[f_])
fig.legend(fontsize=6.3, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(P / "figures" / "fig_neighbors.pdf")
plt.close(fig)

def repel_labels(ax, xs, ys, labels, fontsize=5.6):
    """Greedy placement: for each point pick, among 8 offsets, the one whose text box overlaps least with
    already placed labels and points (display coordinates)."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.c_[xs, ys])
    placed = []
    offs = [(4, 3), (4, -9), (-4, 3), (-4, -9), (0, 7), (0, -12), (8, -3), (-8, -3)]
    for i in np.argsort(-np.asarray(ys)):
        best = None
        for dx, dy in offs:
            t = ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, color=INK2, ha="left" if dx >= 0 else "right")
            bb = t.get_window_extent(r).expanded(1.05, 1.15)
            ov = sum(max(0, min(bb.x1, b.x1) - max(bb.x0, b.x0)) * max(0, min(bb.y1, b.y1) - max(bb.y0, b.y0))
                     for b in placed)
            ov += sum(1e4 for (px, py) in pts if bb.x0 < px < bb.x1 and bb.y0 < py < bb.y1)
            if best is None or ov < best[0]:
                if best is not None:
                    best[1].remove()
                best = (ov, t, bb)
            else:
                t.remove()
        placed.append(best[2])


# leakage vs gap
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, (d, t) in zip(axes, DS.items()):
    e = E[d]
    for k in ENC:
        ax.scatter(100 * e.loc[k, "within_style_artist"], 100 * e.loc[k, "artist_gap"], s=22, color=FAMCOL[FAM[k]], marker=FAMMARK[FAM[k]],
                   edgecolor="white", lw=0.6, zorder=3)
    c = pd.read_csv(F / d / "correlations_primary.csv").set_index("gap").loc["artist_gap"]
    ax.set_title(f"{DSN[d]}: $\\rho$={c.rho:.2f}; partial $\\rho\\mid$IID={c['partial_rho|iid']:.2f} "
                 f"[{c.partial_ci_lo:.2f}, {c.partial_ci_hi:.2f}]", fontsize=7, color=INK)
    ax.set_xlabel("within-style artist probe (chance-normalized bAcc, %)")
    ax.margins(x=0.12, y=0.1)
    repel_labels(ax, list(100 * e["within_style_artist"]), list(100 * e["artist_gap"]), [NICE[k] for k in ENC])
    ax.set_ylabel("Artist gap (bAcc points)")
    ax.grid(color=GRID, lw=0.6, zorder=0)
for f_, c in FAMCOL.items():
    axes[0].scatter([], [], color=c, s=18, marker=FAMMARK[f_], label=FAMNAME[f_])
fig.legend(fontsize=6.3, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(P / "figures" / "fig_leak_gap.pdf")
plt.close(fig)

# schematic of the protocols (cells of one seed)
fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.75))
titles = ["IID", "Artist-D", "Genre-D", "Joint", "Factorial designs"]
for ax, ttl in zip(axes, titles):
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(ttl, fontsize=7.5, color=INK)
    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), 0.94, 0.94, fc="#f1f0ec", ec="none"))
    ax.text(-0.08, 1.5, "train\nartists", fontsize=6.4, ha="right", va="center", color=INK2)
    ax.text(-0.08, 0.5, "test\nartists", fontsize=6.4, ha="right", va="center", color=INK2)
    ax.text(0.5, -0.06, "train g.", fontsize=6.4, ha="center", va="top", color=INK2)
    ax.text(1.5, -0.06, "test g.", fontsize=6.4, ha="center", va="top", color=INK2)
TR, TE, XX = "#2a78d6", "#eb6834", "#d6d4ce"


def cell(ax, x, y, kind, txt=""):
    col = {"train": TR, "test": TE, "drop": XX, "half": None}[kind]
    if kind == "half":
        ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), 0.47, 0.94, fc=TR, ec="none"))
        ax.add_patch(plt.Rectangle((x + 0.5, y + 0.03), 0.47, 0.94, fc=TE, ec="none"))
    else:
        ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), 0.94, 0.94, fc=col, ec="none", alpha=0.85))
    if txt:
        ax.text(x + 0.5, y + 0.5, txt, fontsize=6.6, ha="center", va="center", color="white", fontweight="bold")


for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    cell(axes[0], x, y, "half")
cell(axes[1], 0, 1, "train"); cell(axes[1], 1, 1, "train"); cell(axes[1], 0, 0, "test"); cell(axes[1], 1, 0, "test")
cell(axes[2], 0, 1, "train"); cell(axes[2], 0, 0, "train"); cell(axes[2], 1, 1, "test"); cell(axes[2], 1, 0, "test")
cell(axes[3], 0, 1, "train"); cell(axes[3], 1, 0, "test"); cell(axes[3], 0, 0, "drop"); cell(axes[3], 1, 1, "drop")
cell(axes[4], 0, 1, "train", "always"); cell(axes[4], 1, 1, "train", "if G\nseen")
cell(axes[4], 0, 0, "train", "if A\nseen"); cell(axes[4], 1, 0, "half")
axes[4].text(1.265, 0.5, "A+G", fontsize=6.0, ha="center", va="center", color="white", fontweight="bold")
axes[4].text(1.735, 0.5, "test", fontsize=6.4, ha="center", va="center", color="white", fontweight="bold")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=TR), plt.Rectangle((0, 0), 1, 1, fc=TE), plt.Rectangle((0, 0), 1, 1, fc=XX)],
           labels=["training (random image-level split where both colours share a cell)", "test", "discarded"],
           loc="lower center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.2, wspace=0.55)
fig.savefig(P / "figures" / "fig_protocol_schematic.pdf")
plt.close(fig)

# ranking
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
for ax, d in zip(axes, DS):
    e = E[d]
    ri, rj = e["iid"].rank(ascending=False), e["joint"].rank(ascending=False)
    for k in ENC:
        ax.plot([0, 1], [ri[k], rj[k]], color=FAMCOL[FAM[k]], lw=1.4, marker=FAMMARK[FAM[k]], ms=3.5)
        ax.text(-0.05, ri[k], f"{NICE[k]}  {100*e.loc[k,'iid']:.1f}", ha="right", va="center", fontsize=6.2, color=INK)
        ax.text(1.05, rj[k], f"{100*e.loc[k,'joint']:.1f}  {NICE[k]}", ha="left", va="center", fontsize=6.2, color=INK)
    ax.invert_yaxis(); ax.axis("off")
    ax.set_xlim(-0.9, 1.9)
    mr = json.load(open(F / d / "master_results.json"))["rankings"]
    ax.set_title(f"{DSN[d]}: IID $\\rightarrow$ Joint (Kendall $\\tau$={mr['kendall_tau_iid_vs_joint']:.2f})",
                 fontsize=7.5, color=INK)
fig.tight_layout()
fig.savefig(P / "figures" / "fig_ranking.pdf")
plt.close(fig)
print(f"{len(M)} macros; tables and figures written")


# ======================================================================== revision round 1 assets
R1 = {}


def rmac(name, val):
    assert name.isalpha(), name
    R1[name] = val


rmac("nEnc", str(len(ENC)))
FX = pd.read_csv(F / "factorial" / "effects.csv")
FX = FX[FX.evaluator == "knn5"]
FS = json.load(open(F / "factorial" / "summary.json"))
EXPO = pd.read_csv(F / "factorial" / "exposure.csv")
DSX = {"wikiart": "WA", "multitask_painting": "MP", "mp_novel": "NV"}
for d, t in DSX.items():
    for des, dk in [("mf", "Mf"), ("emf", "Emf")]:
        sm = FS[f"{d}|{des}|knn5"]
        rmac(f"{t}{dk}Gs", num(100 * sm["genre_Aplus_mean"]))
        rmac(f"{t}{dk}Gu", num(100 * sm["genre_Aminus_mean"]))
        rmac(f"{t}{dk}Art", num(100 * sm["artist_mean"]))
        rmac(f"{t}{dk}Int", num(100 * sm["interaction_mean"]))
        rmac(f"{t}{dk}IntPos", str(sm["interaction_pos_hier"]))
        rmac(f"{t}{dk}Equiv", str(sm["genre_Aminus_upper_below_delta_hier"]))
        rmac(f"{t}{dk}GuPos", str(sm["genre_Aminus_pos_hier"]))
        rmac(f"{t}{dk}Ratio", num(100 * sm["ratio_Aminus_over_Aplus"], 0))
        q = FX[(FX.dataset == d) & (FX.design == des) & (FX.effect == "genre_Aminus")]
        rmac(f"{t}{dk}Gulo", num(100 * q["mean"].min())); rmac(f"{t}{dk}Guhi", num(100 * q["mean"].max()))
        q = FX[(FX.dataset == d) & (FX.design == des) & (FX.effect == "genre_Aplus")]
        rmac(f"{t}{dk}Gslo", num(100 * q["mean"].min())); rmac(f"{t}{dk}Gshi", num(100 * q["mean"].max()))
    smn = FS[f"{d}|emf|ncc"]
    rmac(f"{t}EmfNccArt", num(100 * smn["artist_mean"])); rmac(f"{t}EmfNccInt", num(100 * smn["interaction_mean"]))
    rmac(f"{t}EmfNccGs", num(100 * smn["genre_Aplus_mean"]))
    smn = FS[f"{d}|mf|ncc"]
    rmac(f"{t}MfNccIntPos", str(smn["interaction_pos_hier"]))
rmac("intPosHierAll", str(sum(FS[f"{d}|mf|knn5"]["interaction_pos_hier"] for d in DS)))
rmac("emfIntPosAll", str(sum(FS[f"{d}|emf|knn5"]["interaction_pos_hier"] for d in DSX)))
rmac("nPairsAll", str(len(ENC) * 2))
rmac("nPairsEmf", str(len(ENC) * 3))
for d, t in DS.items():
    ex = EXPO[EXPO.dataset == d].groupby(["design", "cond"]).mean(numeric_only=True)
    rmac(f"{t}MfTestArt", num(ex.loc[("mf", "AG"), "test_artists"], 0))
    rmac(f"{t}MfSeenAG", num(ex.loc[("mf", "AG"), "test_artists_seen"], 0))
    rmac(f"{t}MfSeenA", num(ex.loc[("mf", "A"), "test_artists_seen"], 0))
    rmac(f"{t}MfPerAG", num(ex.loc[("mf", "AG"), "train_imgs_per_seen_artist_median"], 1))
    rmac(f"{t}MfPerA", num(ex.loc[("mf", "A"), "train_imgs_per_seen_artist_median"], 1))
    rmac(f"{t}EmfTestArt", num(ex.loc[("emf", "AG"), "test_artists"], 0))
    rmac(f"{t}EmfPer", num(ex.loc[("emf", "AG"), "train_imgs_per_seen_artist_median"], 1))
    er = pd.read_csv(A / "reports" / f"{d}_emf_report.csv")
    rmac(f"{t}EmfAddon", f"{int(er.n_addon.mean()):,}")
    rmac(f"{t}EmfTrain", f"{int(er.n_train.mean()):,}")
    rmac(f"{t}EmfAddonPct", num(100 * (er.n_addon / er.n_train).mean(), 1))
    rmac(f"{t}EmfGenreMatch", num(100 * er.emf_G_genre_matched_share.mean(), 0))
    rmac(f"{t}EmfStyles", num(er.n_styles.mean(), 0))
    # neighbour intervention
    x = pd.read_csv(A / "results" / "neighbor_intervention" / f"{d}.csv")
    x = x[x.k == 5].replace({"condition": {"control_random": "control"}})
    g = x.groupby(["encoder", "condition"]).bacc.mean().unstack() * 100
    rmac(f"{t}IntvNone", num(g["none"].mean())); rmac(f"{t}IntvSame", num(g["same_artist"].mean()))
    rmac(f"{t}IntvCtrl", num(g["control"].mean()))
    dd = g["none"] - g["same_artist"]
    rmac(f"{t}IntvDroplo", num(dd.min())); rmac(f"{t}IntvDrophi", num(dd.max()))
    rmac(f"{t}IntvCtrlmax", num((g["none"] - g["control"]).max()))
    rmac(f"{t}IntvShare", num(100 * x.share_with_same_artist.mean(), 1))
    rmac(f"{t}IntvRemoved", num(x.mean_removed.mean(), 0))
    # duplicate audit
    fl = pd.read_csv(A / "results" / "dup_audit" / f"{d}_flags.csv").groupby("protocol").mean(numeric_only=True)
    ef = pd.read_csv(A / "results" / "dup_audit" / f"{d}_effect.csv")
    ef["delta"] = 100 * (ef.bacc_unflagged - ef.bacc_all)
    de = ef.groupby(["protocol", "encoder"]).delta.mean()
    rmac(f"{t}DupIidPct", num(100 * fl.loc["iid", "flag_union"] / fl.loc["iid", "n_test"], 1))
    rmac(f"{t}DupArtPct", num(100 * fl.loc["artist_disjoint", "flag_union"] / fl.loc["artist_disjoint", "n_test"], 1))
    rmac(f"{t}DupIidMax", num(de.loc["iid"].abs().max(), 1))
    rmac(f"{t}DupCtrlMax", num(de.drop("iid").abs().max(), 1))
    # algorithm diagnostics
    dg = pd.read_csv(A / "reports" / "split_optimizer_diagnostics.csv")
    dg = dg[dg.d == d]
    rmac(f"AlgDevArt{t}", num(dg[dg.p == "artist_disjoint"].absdev_mean.mean(), 3))
    rmac(f"AlgDevJoint{t}", num(dg[dg.p == "joint"].absdev_mean.mean(), 3))
ov = json.load(open(A / "reports" / "cross_dataset_overlap.json"))
rmac("OvWAshare", num(100 * ov["cleaned"]["share_wikiart"], 0))
rmac("OvMPshare", num(100 * ov["cleaned"]["share_mp"], 0))
rmac("OvArtShare", num(100 * ov["cleaned"]["share_wikiart_artists_in_mp"], 0))
rmac("OvArtShared", f"{ov['cleaned']['artists_shared']:,}")
rmac("OvPoolWAshare", num(100 * ov["controlled_pool"]["share_wikiart"], 0))
rmac("OvPoolMPshare", num(100 * ov["controlled_pool"]["share_mp"], 0))
nvc = pd.read_parquet(A / "metadata" / "mp_novel" / "cleaned.parquet")
rmac("NVimages", f"{len(nvc):,}")
nvs = pd.read_csv(A / "splits" / "mp_novel" / "split_summary.csv")
nvcore = pd.read_csv(A / "splits" / "mp_novel" / "core_style_table.csv", index_col=0)
rmac("NVcore", str(int((nvcore.excluded_reason.fillna("") == "").sum())))
rmac("NVpool", f"{int(nvs[(nvs.protocol=='iid')&(nvs.seed==0)][['n_train_total','n_test']].sum(1).iloc[0]):,}")
VAR = {"wikiart": "WikiArt (main)", "wikiart_phash8": "WikiArt, pHash $\\le$ 8",
       "wikiart_keepconf": "WikiArt, keep one copy", "multitask_painting": "MultitaskPainting100k (main)",
       "multitask_painting_phash8": "MultitaskPainting100k, pHash $\\le$ 8",
       "multitask_painting_keepconf": "MultitaskPainting100k, keep one copy",
       "mp_novel": "MultitaskPainting100k without WikiArt duplicates"}
vk = {"wikiart": "WA", "wikiart_phash8": "WAph", "wikiart_keepconf": "WAkc", "multitask_painting": "MP",
      "multitask_painting_phash8": "MPph", "multitask_painting_keepconf": "MPkc", "mp_novel": "NV"}
for v, t in vk.items():
    ev = pd.read_csv(F / v / "encoder_level.csv", index_col=0)
    for p, k in PRO.items():
        rmac(f"{t}v{k}", num(100 * ev[p].mean()))
    rmac(f"{t}vGs", num(100 * (ev.mf_iid - ev.mf_genre).mean()))
    rmac(f"{t}vGu", num(100 * (ev.mf_artist - ev.mf_joint).mean()))
    cs = json.load(open(A / "metadata" / v / "cleaning_stats.json"))
    rmac(f"{t}vN", f"{cs['n_after_clean']:,}")
# MP dropped styles (seed 1)
with open(P / "numbers.tex", "a") as f:
    f.write("% ---- revision round 1\n")
    for k in sorted(R1):
        f.write(f"\\newcommand{{\\{k}}}{{{R1[k]}}}\n")
M.update(R1)
json.dump(M, open(P / "numbers.json", "w"), indent=0)

# ---- table: factorial effects per dataset (MF and EMF, kNN-5, hierarchical 95% intervals)
EFN = {"artist": "$\\Delta_{\\mathrm{A}}$", "genre_Aplus": "$\\Delta_{\\mathrm{G}\\mid\\mathrm{A}^{+}}$",
       "genre_Aminus": "$\\Delta_{\\mathrm{G}\\mid\\mathrm{A}^{-}}$", "interaction": "$I$"}
for d, t in DS.items():
    lines = [r"\begin{tabular}{l cccc ccc}", r"\toprule",
             r" & \multicolumn{4}{c}{Matched factorial} & \multicolumn{3}{c}{Exposure-matched factorial} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
             "Encoder & " + " & ".join(EFN[k] for k in ["artist", "genre_Aplus", "genre_Aminus", "interaction"]) + " & "
             + " & ".join(EFN[k] for k in ["genre_Aplus", "genre_Aminus", "interaction"]) + r" \\", r"\midrule"]
    for e in ENC:
        cells = []
        for des, ks in [("mf", ["artist", "genre_Aplus", "genre_Aminus", "interaction"]),
                        ("emf", ["genre_Aplus", "genre_Aminus", "interaction"])]:
            for k in ks:
                r = FX[(FX.dataset == d) & (FX.encoder == e) & (FX.design == des) & (FX.effect == k)].iloc[0]
                if k in ("genre_Aminus", "interaction"):
                    cells.append(f"{100*r['mean']:.1f} {{\\scriptsize[{100*r.hier_lo:.1f}, {100*r.hier_hi:.1f}]}}")
                else:
                    cells.append(f"{100*r['mean']:.1f}")
        lines.append(NICE[e] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (P / "tables" / f"tab_fact_{t}.tex").write_text("\n".join(lines) + "\n")

# ---- table: exposure of the test artists in each condition
CN = {"AG": "A and G seen", "A": "A seen, G unseen", "G": "G seen, A unseen", "none": "neither"}
lines = [r"\begin{tabular}{ll l rrr r}", r"\toprule",
         r"Dataset & Design & Condition & Test artists & seen in training & img./seen artist & test-genre img. \\", r"\midrule"]
for d in DS:
    ex = EXPO[EXPO.dataset == d].groupby(["design", "cond"]).mean(numeric_only=True)
    first = True
    for des, dn in [("mf", "Matched"), ("emf", "Exposure-matched")]:
        for c in ["AG", "A", "G", "none"]:
            r = ex.loc[(des, c)]
            lines.append((DSN[d] if first else "") + f" & {dn if c == 'AG' else ''} & {CN[c]} & {r.test_artists:.0f} & "
                         f"{r.test_artists_seen:.0f} & {r.train_imgs_per_seen_artist_median:.1f} & {r.test_genre_imgs_in_train:,.0f} \\\\")
            first = False
    if d == "wikiart":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_exposure.tex").write_text("\n".join(lines) + "\n")

# ---- corrected split statistics + cell manifest
lines = [r"\begin{tabular}{l l rrrr r r l}", r"\toprule",
         r"Dataset & Protocol & Fit & Val & Sel.-excl. & Train total & Test & Test artists & Overlap (A / G) \\", r"\midrule"]
for d in DS:
    ss = pd.read_csv(A / "splits" / d / "split_summary.csv")
    for i, p in enumerate(pn):
        q = ss[ss.protocol == p]
        lines.append((DSN[d] if i == 0 else "") + f" & {pn[p]} & {round(q.n_train_fit.mean()):,} & {round(q.n_val.mean()):,} & "
                     f"{round(q.n_sel_excluded.mean()):,} & {round(q.n_train_total.mean()):,} & {round(q.n_test.mean()):,} & "
                     f"{round(q.n_test_artists.mean())} & {q.artist_overlap.mean():.0f} / {q.genre_overlap.mean():.0f} \\\\")
    if d == "wikiart":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_splits.tex").write_text("\n".join(lines) + "\n")

sys.path.insert(0, str(ROOT / "src"))
from stylebench.config import split_config  # noqa: E402
from stylebench.data.splits import controlled_pool  # noqa: E402
cells_rows = []
for d in DS:
    clean = pd.read_parquet(A / "metadata" / d / "cleaned.parquet")
    pool, _ = controlled_pool(clean, split_config()["support"])
    for seed in range(10):
        jd = json.load(open(A / "splits" / d / "joint" / f"seed{seed}.json"))
        styles = set(pd.read_parquet(A / "splits" / d / "joint" / f"seed{seed}.parquet")["style"])
        pp = pool[pool["style"].isin(styles)]
        am, gm = pp["artist"].isin(jd["test_artists"]), pp["genre"].isin(jd["test_genres"])
        cnt = {"BB": int((~am & ~gm).sum()), "AT": int((~am & gm).sum()), "TA": int((am & ~gm).sum()), "TT": int((am & gm).sum())}
        jp = pd.read_parquet(A / "splits" / d / "joint" / f"seed{seed}.parquet")
        mfp = pd.read_parquet(A / "splits" / d / "mf_joint" / f"seed{seed}.parquet")
        jt, jte, mft = int((jp.part != "test").sum()), int((jp.part == "test").sum()), int((mfp.part != "test").sum())
        assert jt == cnt["BB"] and jte == cnt["TT"] and mft <= cnt["BB"], (d, seed)   # checked for every seed
        assert set(mfp[mfp.part != "test"].image_id) <= set(jp[jp.part != "test"].image_id)
        cells_rows.append({"d": d, "seed": seed, **cnt, "jt": jt, "jte": jte, "mft": mft, "disc": cnt["AT"] + cnt["TA"]})
CR = pd.DataFrame(cells_rows)
lines = [r"\begin{tabular}{l l rrrr rrr r}", r"\toprule",
         r"Dataset & & BB & AT & TA & TT & Joint train & Joint test & MF train & Discarded \\", r"\midrule"]
for d in DS:
    z = CR[CR.d == d]
    cols = ["BB", "AT", "TA", "TT", "jt", "jte", "mft", "disc"]
    lines.append(f"{DSN[d]} & mean & " + " & ".join(f"{round(z[c].mean()):,}" for c in cols) + r" \\")
    lines.append(" & min--max & " + " & ".join(f"{z[c].min():,}--{z[c].max():,}" for c in cols) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_cells.tex").write_text("\n".join(lines) + "\n")

# ---- replication / sensitivity table
lines = [r"\begin{tabular}{l r cccc cc}", r"\toprule",
         r"Data & Images & IID & Artist-D & Genre-D & Joint & $\Delta_{\mathrm{G}\mid\mathrm{A}^{+}}$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ \\", r"\midrule"]
for v, t in vk.items():
    lines.append(f"{VAR[v]} & {R1[t+'vN']} & " + " & ".join(R1[f"{t}v{k}"] for k in ["Iid", "Art", "Gen", "Joint"])
                 + f" & {R1[t+'vGs']} & {R1[t+'vGu']} \\\\")
    if v in ("wikiart_keepconf", "multitask_painting_keepconf"):
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_replication.tex").write_text("\n".join(lines) + "\n")

# ---- per-style table with EMF genre effects
pse = pd.read_csv(F / "factorial" / "perstyle_effects.csv")
pse = pse[(pse.evaluator == "knn5") & (pse.design == "emf")].groupby(["dataset", "style", "effect"]).value.mean().unstack()
ps = pd.read_csv(F / "per_style_recall_gaps.csv")
lines = [r"\begin{tabular}{l l r r c c cc}", r"\toprule",
         r"Dataset & Style & Images & Artists & IID recall & Artist gap & EMF $\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ & EMF $I$ \\", r"\midrule"]
for d in DS:
    q = ps[ps.dataset == d].sort_values("artist_gap", ascending=False)
    for i, (_, r) in enumerate(q.iterrows()):
        key = (d, r["style"])
        gu = f"{100*pse.loc[key,'genre_Aminus']:.1f}" if key in pse.index else "--"
        it = f"{100*pse.loc[key,'interaction']:.1f}" if key in pse.index else "--"
        lines.append((DSN[d] if i == 0 else "") + f" & {r['style'].title()} & {int(r.n_images):,} & {int(r.n_artists)} & "
                     f"{100*r.iid:.1f} & {100*r.artist_gap:.1f} & {gu} & {it} \\\\")
    if d == "wikiart":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_perstyle.tex").write_text("\n".join(lines) + "\n")
pq = pse.reset_index()
for d, t in DS.items():
    z = pq[pq.dataset == d]
    rmac(f"{t}PsGuMax", num(100 * z.genre_Aminus.max())); rmac(f"{t}PsGuMin", num(100 * z.genre_Aminus.min()))
    rmac(f"{t}PsGuMaxStyle", z.loc[z.genre_Aminus.idxmax(), "style"].title())
    rmac(f"{t}PsIntNeg", str(int((z.interaction < 0).sum()))); rmac(f"{t}PsN", str(len(z)))
with open(P / "numbers.tex", "a") as f:
    for k in ["WAPsGuMax", "WAPsGuMin", "WAPsGuMaxStyle", "WAPsIntNeg", "WAPsN", "MPPsGuMax", "MPPsGuMin",
              "MPPsGuMaxStyle", "MPPsIntNeg", "MPPsN"]:
        f.write(f"\\newcommand{{\\{k}}}{{{R1[k]}}}\n")
M.update(R1)
json.dump(M, open(P / "numbers.json", "w"), indent=0)

# ---- figure: neighbour intervention (k = 5; four conditions; round 2 adds the rank-matched control)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
cond = [("none", "all training\npaintings"), ("control_random", "remove random\nsame-style,\nother-artist"),
        ("control_nearest", "remove nearest\nsame-style,\nother-artist"), ("same_artist", "remove\nsame-artist\npaintings")]
for ax, d in zip(axes, DS):
    x = pd.read_csv(A / "results" / "neighbor_intervention" / f"{d}.csv")
    x = x[x.k == 5]
    g = x.groupby(["encoder", "condition"]).bacc.mean().unstack().loc[ENC] * 100
    for e in ENC:
        ax.plot(range(4), [g.loc[e, c] for c, _ in cond], color=FAMCOL[FAM[e]], lw=1.0, marker=FAMMARK[FAM[e]], ms=3, alpha=0.9)
    for i_, (c, _) in enumerate(cond):
        ax.text(i_, g[c].max() + 3, f"mean {g[c].mean():.1f}", ha="center", fontsize=6.6, color=INK)
    ax.set_xticks(range(4), [c[1] for c in cond], fontsize=6.2)
    ax.set_ylabel("balanced accuracy, IID test set (%)")
    ax.set_title(DSN[d], fontsize=8, color=INK)
    ax.set_ylim(0, 90)
    ax.grid(axis="y", color=GRID, lw=0.6)
for f_, c in FAMCOL.items():
    axes[0].plot([], [], color=c, lw=1.4, marker=FAMMARK[f_], label=FAMNAME[f_])
fig.legend(fontsize=6.3, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(P / "figures" / "fig_intervention.pdf")
plt.close(fig)
print(f"revision: {len(R1)} macros; factorial/exposure/cells/replication/perstyle tables; intervention figure")

# ---- per-style counts of sizeable genre effects without artist exposure (EMF, kNN-5)
with open(P / "numbers.tex", "a") as f:
    for d, t in DS.items():
        z = pq[pq.dataset == d].set_index("style")
        big = z.index[100 * z.genre_Aminus > 5]
        R1[f"{t}PsGuBigN"] = str(len(big))
        R1[f"{t}PsGuBigList"] = ", ".join(s.title() for s in sorted(big, key=lambda s: -z.loc[s, "genre_Aminus"]))
        R1[f"{t}PsGuMaxVal"] = num(100 * z.genre_Aminus.max())
        for k in [f"{t}PsGuBigN", f"{t}PsGuBigList", f"{t}PsGuMaxVal"]:
            f.write(f"\\newcommand{{\\{k}}}{{{R1[k]}}}\n")
M.update(R1)
json.dump(M, open(P / "numbers.json", "w"), indent=0)


# ======================================================================== revision round 2 assets
R2 = {}
RV = A / "results" / "rev2"


def m2(name, val):
    assert name.isalpha(), name
    R2[name] = val


FA = pd.read_csv(F / "factorial" / "effects.csv")
FSUM = json.load(open(F / "factorial" / "summary.json"))
FDED = json.load(open(F / "factorial_dedup" / "summary.json")) if (F / "factorial_dedup" / "summary.json").exists() else {}
EFFK = ["artist", "genre_Aplus", "genre_Aminus", "interaction"]
EFN2 = {"artist": "$\\Delta_{\\mathrm{A}}$", "genre_Aplus": "$\\Delta_{\\mathrm{G}\\mid\\mathrm{A}^{+}}$",
        "genre_Aminus": "$\\Delta_{\\mathrm{G}\\mid\\mathrm{A}^{-}}$", "interaction": "$I$"}


def eff_table(dataset, design, fname):
    q = FA[(FA.dataset == dataset) & (FA.design == design) & (FA.evaluator == "knn5")]
    lines = [r"\begin{tabular}{l cccc}", r"\toprule", "Encoder & " + " & ".join(EFN2[k] for k in EFFK) + r" \\", r"\midrule"]
    for e in ENC:
        cells = []
        for k in EFFK:
            r = q[(q.encoder == e) & (q.effect == k)].iloc[0]
            cells.append(f"{f1(100*r['mean'])} {{\\scriptsize[{f1(100*r.bca_lo)}, {f1(100*r.bca_hi)}]}}")
        lines.append(NICE[e] + " & " + " & ".join(cells) + r" \\")
    mean = [f1(100 * q[q.effect == k]["mean"].mean()) for k in EFFK]
    lines += [r"\midrule", "Mean & " + " & ".join(mean) + r" \\", r"\bottomrule", r"\end{tabular}"]
    (P / "tables" / f"{fname}.tex").write_text("\n".join(lines) + "\n")


for d, t in DS.items():
    eff_table(d, "emf", f"tab_emf_{t}")
    eff_table(d, "mf", f"tab_mfapp_{t}")


def summ(key, src=None):
    return (src or FSUM).get(key)


# design robustness table
VARS = [("Exposure-matched", "emf", "wikiart", "knn5", "WikiArt", None), ("", "emf", "multitask_painting", "knn5", "MultitaskPainting100k", None),
        ("", "emf", "wikiart_ga", "knn5", "WikiArt, genre-agnostic partition", None),
        ("", "emf", "multitask_painting_ga", "knn5", "MultitaskPainting100k, genre-agnostic partition", None),
        ("", "emf", "wikiart", "knn5", "WikiArt, near-duplicates excluded", "dedup"),
        ("", "emf", "multitask_painting", "knn5", "MultitaskPainting100k, near-duplicates excluded", "dedup"),
        ("", "emf", "mp_novel", "knn5", "MultitaskPainting100k without a WikiArt counterpart", None),
        ("", "emf", "wikiart", "logreg", "WikiArt, logistic regression", None),
        ("", "emf", "multitask_painting", "logreg", "MultitaskPainting100k, logistic regression", None),
        ("", "emf", "wikiart", "ncc", "WikiArt, nearest centroid", None),
        ("", "emf", "multitask_painting", "ncc", "MultitaskPainting100k, nearest centroid", None),
        ("Matched", "mf", "wikiart", "knn5", "WikiArt", None), ("", "mf", "multitask_painting", "knn5", "MultitaskPainting100k", None),
        ("", "mf", "wikiart_ga", "knn5", "WikiArt, genre-agnostic partition", None),
        ("", "mf", "multitask_painting_ga", "knn5", "MultitaskPainting100k, genre-agnostic partition", None),
        ("", "mf", "wikiart", "ncc", "WikiArt, nearest centroid", None), ("", "mf", "multitask_painting", "ncc", "MultitaskPainting100k, nearest centroid", None)]
lines = [r"\begin{tabular}{l l c ccc c c}", r"\toprule",
         r"Design & Data / classifier & Enc. & $\Delta_{\mathrm{G}\mid\mathrm{A}^{+}}$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ & $I$ & $I>0$ (BCa) & max upper $\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ \\", r"\midrule"]
for lab, des, d, ev, name, src in VARS:
    s_ = summ(f"{d}|{des}|{ev}", FDED if src == "dedup" else None)
    if lab == "Matched":
        lines.append(r"\midrule")
    if s_ is None or s_["n_encoders"] < len(ENC):
        cells = ["--"] * 6
    else:
        cells = [str(s_["n_encoders"]), f1(100 * s_["genre_Aplus_mean"]), f1(100 * s_["genre_Aminus_mean"]),
                 f1(100 * s_["interaction_mean"]), f"{s_['interaction_pos_bca']}/{s_['n_encoders']}",
                 f1(100 * s_["genre_Aminus_max_upper_bca"])]
    lines.append(f"{lab} & {name} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_fact_robust.tex").write_text("\n".join(lines) + "\n")
for (d, des, ev, src), t in {("wikiart", "emf", "knn5", None): "WAemf", ("multitask_painting", "emf", "knn5", None): "MPemf",
                             ("wikiart_ga", "emf", "knn5", None): "WAgaemf", ("multitask_painting_ga", "emf", "knn5", None): "MPgaemf",
                             ("wikiart_ga", "mf", "knn5", None): "WAgamf", ("multitask_painting_ga", "mf", "knn5", None): "MPgamf",
                             ("wikiart", "mf", "knn5", None): "WAmf", ("multitask_painting", "mf", "knn5", None): "MPmf",
                             ("wikiart", "emf", "knn5", "dedup"): "WAdedemf", ("multitask_painting", "emf", "knn5", "dedup"): "MPdedemf",
                             ("wikiart", "emf", "ncc", None): "WAemfncc", ("multitask_painting", "emf", "ncc", None): "MPemfncc",
                             ("wikiart", "emf", "logreg", None): "WAemflr", ("multitask_painting", "emf", "logreg", None): "MPemflr",
                             ("mp_novel", "emf", "knn5", None): "NVemf"}.items():
    s_ = summ(f"{d}|{des}|{ev}", FDED if src == "dedup" else None)
    if s_ is None:
        continue
    m2(f"{t}Gs", f1(100 * s_["genre_Aplus_mean"])); m2(f"{t}Gu", f1(100 * s_["genre_Aminus_mean"]))
    m2(f"{t}I", f1(100 * s_["interaction_mean"])); m2(f"{t}Art", f1(100 * s_["artist_mean"]))
    m2(f"{t}Ipos", str(s_["interaction_pos_bca"])); m2(f"{t}N", str(s_["n_encoders"]))
    m2(f"{t}Gumax", f1(100 * s_["genre_Aminus_max_upper_bca"])); m2(f"{t}Gupos", str(s_["genre_Aminus_pos_bca"]))
    m2(f"{t}Bias", f1(100 * s_["interaction_bias_mean"]))
    m2(f"{t}Equiv", str(s_["genre_Aminus_upper_below_delta_bca"]))
# bootstrap diagnostics (C4)
q = FA[(FA.evaluator == "knn5") & FA.dataset.isin(["wikiart", "multitask_painting"])]
bi = q[q.effect == "interaction"].boot_bias
assert (bi < 0).all()
m2("BiasIntMin", f1(100 * bi.abs().min())); m2("BiasIntMax", f1(100 * bi.abs().max()))
m2("BiasGuMax", f1(100 * q[q.effect == "genre_Aminus"].boot_bias.abs().max()))
m2("DropRepMax", f1(100 * q.share_style_drops_per_seed_rep.max(), 0))
nd = max((q.alt_nan_lo - q.hier_lo).abs().max(), (q.alt_nan_hi - q.hier_hi).abs().max())
m2("NodropShift", f1(100 * nd))

# scales (C1)
SC = pd.read_csv(F / "factorial" / "effects_scales.csv")
SC = SC[(SC.evaluator == "knn5") & SC.dataset.isin(["wikiart", "multitask_painting"])]
lines = [r"\begin{tabular}{l l ccc ccc ccc c}", r"\toprule",
         r" & & \multicolumn{3}{c}{bAcc points} & \multicolumn{3}{c}{logit} & \multicolumn{3}{c}{relative (\%)} & \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"Dataset & Design & $\Delta_{\mathrm{G}\mid\mathrm{A}^{+}}$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ & $I$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{+}}$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ & $I$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{+}}$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ & $I$ & $I>0$ (logit) \\", r"\midrule"]
for d in DS:
    for des, dn in [("emf", "Exposure-matched"), ("mf", "Matched")]:
        z = SC[(SC.dataset == d) & (SC.design == des)]
        g = z.groupby("effect")[["bacc_pts", "logit", "relative_pct"]].mean()
        ip = int((z[z.effect == "interaction"].logit > 0).sum())
        lines.append(f"{DSN[d] if des == 'emf' else ''} & {dn} & " + " & ".join(
            [f1(g.loc[k, "bacc_pts"]) for k in ["genre_Aplus", "genre_Aminus", "interaction"]] +
            [f1(g.loc[k, "logit"], 2) for k in ["genre_Aplus", "genre_Aminus", "interaction"]] +
            [f1(g.loc[k, "relative_pct"]) for k in ["genre_Aplus", "genre_Aminus", "interaction"]]) + f" & {ip}/{len(z[z.effect=='interaction'])} \\\\")
        if des == "emf":
            m2(f"{DS[d]}emfLogitGs", f1(g.loc['genre_Aplus', 'logit'], 2)); m2(f"{DS[d]}emfLogitGu", f1(g.loc['genre_Aminus', 'logit'], 2))
            m2(f"{DS[d]}emfRelGs", f1(g.loc['genre_Aplus', 'relative_pct'], 0)); m2(f"{DS[d]}emfRelGu", f1(g.loc['genre_Aminus', 'relative_pct'], 0))
            m2(f"{DS[d]}emfLogitIpos", str(ip))
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_scales.tex").write_text("\n".join(lines) + "\n")

# decomposition + size control + zero-shot (C5, D2)
DC = pd.read_csv(RV / "decomposition.csv")
SZ = pd.read_csv(RV / "size_control.csv")
ZS = pd.read_csv(RV / "zero_shot.csv")
VLE = ["clip_b16", "clip_l14", "openclip_b16", "siglip_b16", "siglip2_l16", "siglip_so400m"]
lines = [r"\begin{tabular}{l l cc}", r"\toprule", r"Analysis & Quantity & WikiArt & MultitaskPainting100k \\", r"\midrule"]
dm = DC.groupby("dataset").mean(numeric_only=True) * 100
sm = SZ.groupby("dataset").mean(numeric_only=True) * 100
items = [("Artist-D $\\rightarrow$ Joint", "Artist-D (standard)", lambda d: dm.loc[d, "A_artistD"]),
         ("", "train BB$\\cup$AT, test TT", lambda d: dm.loc[d, "B_jointpart_full"]),
         ("", "same, training size matched to BB", lambda d: dm.loc[d, "C_jointpart_sizematched"]),
         ("", "Joint (standard; train BB, test TT)", lambda d: dm.loc[d, "D_joint"]),
         ("", "component: test set and partition", lambda d: dm.loc[d, "A_artistD"] - dm.loc[d, "B_jointpart_full"]),
         ("", "component: training size", lambda d: dm.loc[d, "B_jointpart_full"] - dm.loc[d, "C_jointpart_sizematched"]),
         ("", "component: genre exposure", lambda d: dm.loc[d, "C_jointpart_sizematched"] - dm.loc[d, "D_joint"]),
         ("Training size", "IID", lambda d: sm.loc[d, "iid"]),
         ("", "IID, subsampled to Genre-D size", lambda d: sm.loc[d, "iid_at_genre_size"]),
         ("", "IID, subsampled to Joint size", lambda d: sm.loc[d, "iid_at_joint_size"]),
         ("", "Artist-D, subsampled to Joint size", lambda d: sm.loc[d, "artist_at_joint_size"])]
for lab, qn, fn in items:
    lines.append(f"{lab} & {qn} & " + " & ".join(f1(fn(d)) for d in DS) + r" \\")
zs = ZS.groupby(["dataset", "protocol"]).bacc.mean().unstack() * 100
kn = {d: pd.read_csv(F / d / "encoder_level.csv", index_col=0).loc[VLE] * 100 for d in DS}
lines.append(r"\midrule")
for p, pn_ in [("iid", "IID"), ("artist_disjoint", "Artist-D"), ("genre_disjoint", "Genre-D"), ("joint", "Joint")]:
    lines.append(("Zero-shot vs.\\ $k$NN (6 VL enc.)" if p == "iid" else "") + f" & {pn_}: zero-shot / $k$NN & " +
                 " & ".join(f"{f1(zs.loc[d, p])} / {f1(kn[d][p].mean())}" for d in DS) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_decomp.tex").write_text("\n".join(lines) + "\n")
for d, t in DS.items():
    m2(f"{t}DecTest", f1(dm.loc[d, "A_artistD"] - dm.loc[d, "B_jointpart_full"]))
    m2(f"{t}DecSize", f1(dm.loc[d, "B_jointpart_full"] - dm.loc[d, "C_jointpart_sizematched"]))
    m2(f"{t}DecGenre", f1(dm.loc[d, "C_jointpart_sizematched"] - dm.loc[d, "D_joint"]))
    m2(f"{t}DecTotal", f1(dm.loc[d, "A_artistD"] - dm.loc[d, "D_joint"]))
    m2(f"{t}SizeIidJoint", f1(sm.loc[d, "iid"] - sm.loc[d, "iid_at_joint_size"]))
    m2(f"{t}SizeIidGenre", f1(sm.loc[d, "iid"] - sm.loc[d, "iid_at_genre_size"]))
    m2(f"{t}SizeArtJoint", f1(sm.loc[d, "artist"] - sm.loc[d, "artist_at_joint_size"]))
    for p, k in [("iid", "Iid"), ("artist_disjoint", "Art"), ("genre_disjoint", "Gen"), ("joint", "Joint")]:
        m2(f"{t}Zs{k}", f1(zs.loc[d, p])); m2(f"{t}ZsKnn{k}", f1(kn[d][p].mean()))
    zso = ZS[(ZS.dataset == d) & (ZS.encoder == "siglip_so400m")].groupby("protocol").bacc.mean() * 100
    m2(f"{t}ZsSoArt", f1(zso["artist_disjoint"])); m2(f"{t}ZsSoIid", f1(zso["iid"]))
    m2(f"{t}ZsRange", f1((zs.loc[d].max() - zs.loc[d].min())))

# intervention k sweep (C8, C9)
lines = [r"\begin{tabular}{l r ccccc}", r"\toprule",
         r"Dataset & $k$ & all & random control & nearest control & same artist (same style) & same artist (any style) \\", r"\midrule"]
for d, t in DS.items():
    x = pd.read_csv(A / "results" / "neighbor_intervention" / f"{d}.csv")
    g = x.groupby(["k", "condition"]).bacc.mean().unstack() * 100
    for i_, k in enumerate(sorted(g.index)):
        lines.append((DSN[d] if i_ == 0 else "") + f" & {k} & " + " & ".join(f1(g.loc[k, c]) for c in
                     ["none", "control_random", "control_nearest", "same_artist_style", "same_artist"]) + r" \\")
    x5 = x[x.k == 5].groupby(["encoder", "condition"]).bacc.mean().unstack() * 100
    dn = x5["none"] - x5["control_nearest"]
    ds_ = x5["none"] - x5["same_artist"]
    m2(f"{t}IntvNearlo", f1(dn.min())); m2(f"{t}IntvNearhi", f1(dn.max())); m2(f"{t}IntvNearMean", f1(dn.mean()))
    m2(f"{t}IntvSameMean", f1(ds_.mean())); m2(f"{t}IntvSameGtNear", str(int((ds_ > dn).sum())))
    m2(f"{t}IntvRemovedAll", f1(x.mean_removed.mean(), 0))
    g20 = g.loc[20]
    m2(f"{t}IntvNearKtw", f1(g20["none"] - g20["control_nearest"])); m2(f"{t}IntvSameKtw", f1(g20["none"] - g20["same_artist"]))
    m2(f"{t}IntvStyleOnly", f1((x5["none"] - x5["same_artist_style"]).mean()))
    if d == "wikiart":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_intv_k.tex").write_text("\n".join(lines) + "\n")

# selection bias (C2)
SB = pd.read_csv(RV / "selection_bias.csv")
lines = [r"\begin{tabular}{l l ccc}", r"\toprule",
         r"Dataset & Artist partition & pool & test artists & training artists \\", r"\midrule"]
for d, t in DS.items():
    for part, pn_ in [("joint", "Joint (optimized with genres)"), ("genre_agnostic", "Artist-D (genre-agnostic)")]:
        z = SB[(SB.dataset == d) & (SB.partition == part)]
        lines.append((DSN[d] if part == "joint" else "") + f" & {pn_} & " + " & ".join(
            f"{100*z[c].mean():.1f} ({100*z[c].min():.1f}--{100*z[c].max():.1f})" for c in
            ["pool_heldout_genre_share", "test_artists_heldout_share", "train_artists_heldout_share"]) + r" \\")
        key = "Jt" if part == "joint" else "Ga"
        m2(f"{t}Sel{key}Test", f1(100 * z.test_artists_heldout_share.mean(), 0)); m2(f"{t}Sel{key}Train", f1(100 * z.train_artists_heldout_share.mean(), 0))
        m2(f"{t}SelPool", f1(100 * z.pool_heldout_genre_share.mean(), 0))
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_selection.tex").write_text("\n".join(lines) + "\n")
SBA = pd.read_csv(RV / "selection_bias_artists.csv")
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.1), sharey=False)
for ax, (d, part) in zip(axes, [(d, p) for d in DS for p in ["joint", "genre_agnostic"]]):
    z = SBA[(SBA.dataset == d) & (SBA.partition == part)]
    bins = np.linspace(0, 1, 11)
    ax.hist(z[z.is_test].heldout_share, bins=bins, density=True, color=FAMCOL["ssl"], alpha=0.75, label="test artists")
    ax.hist(z[~z.is_test].heldout_share, bins=bins, density=True, color=FAMCOL["vl"], alpha=0.55, label="training artists")
    ax.set_title(f"{'WikiArt' if d == 'wikiart' else 'MP100k'}, {'Joint' if part == 'joint' else 'genre-agnostic'}", fontsize=7)
    ax.tick_params(labelsize=6)
axes[0].legend(fontsize=5.8, frameon=False)
axes[0].set_ylabel("density", fontsize=7)
fig.supxlabel("share of an artist's paintings in held-out genres", fontsize=7)   # one shared axis title (R3)
fig.tight_layout(w_pad=1.6)
fig.savefig(P / "figures" / "fig_selection.pdf")
plt.close(fig)

# leakage re-analysis (C3)
LR_ = pd.read_csv(RV / "leakage_reanalysis.csv")
lines = [r"\begin{tabular}{l l l c ccc}", r"\toprule",
         r"Dataset & Leakage $\rightarrow$ gap & Classifier & $\rho$ [95\% CI] & partial $\rho\mid$IID & partial $\rho\mid$Joint & partial $\rho\mid$style probe \\", r"\midrule"]
LN = {"within_style_artist": "Artist$\\mid$style", "within_style_genre": "Genre$\\mid$style", "combined_within_style": "Combined"}
for d in DS:
    z = LR_[LR_.dataset == d]
    for i_, (_, r) in enumerate(z.iterrows()):
        pc = lambda k: f"{r[f'partial_{k}']:.2f} [{r[f'partial_{k}_lo']:.2f}, {r[f'partial_{k}_hi']:.2f}]"  # noqa: E731
        lines.append((DSN[d] if i_ == 0 else "") + f" & {LN[r.leak]} & {'$k$NN' if r.classifier == 'knn' else 'logistic'} & "
                     f"{r.rho:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}] & {pc('iid')} & {pc('joint')} & {pc('style')} \\\\")
    if d == "wikiart":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_leak2.tex").write_text("\n".join(lines) + "\n")
for d, t in DS.items():
    z = LR_[LR_.dataset == d].set_index(["leak", "classifier"])
    r = z.loc[("within_style_artist", "logreg")]; m2(f"{t}LrRhoA", f1(r.rho, 2))
    r = z.loc[("within_style_artist", "knn")]; m2(f"{t}PrhoAJoint", f1(r.partial_joint, 2)); m2(f"{t}PrhoAJointlo", f1(r.partial_joint_lo, 2)); m2(f"{t}PrhoAJointhi", f1(r.partial_joint_hi, 2))
    r = z.loc[("within_style_genre", "knn")]; m2(f"{t}PrhoGJoint", f1(r.partial_joint, 2)); m2(f"{t}PrhoGJointlo", f1(r.partial_joint_lo, 2)); m2(f"{t}PrhoGJointhi", f1(r.partial_joint_hi, 2))

# Ashton replication (C10)
AS = pd.read_csv(RV / "ashton_replication.csv")
g = AS.groupby(["movement", "protocol"]).acc.mean().unstack() * 100
lines = [r"\begin{tabular}{l cc c}", r"\toprule", r"Movement & full pool & leave-one-artist-out & drop \\", r"\midrule"]
for mv in ["Abstract Expressionism", "Cubism", "Impressionism", "Surrealism", "all"]:
    lines.append(f"{mv if mv != 'all' else 'All (320 images)'} & {f1(g.loc[mv,'full_pool'])} & {f1(g.loc[mv,'leave_one_artist_out'])} & {f1(g.loc[mv,'full_pool']-g.loc[mv,'leave_one_artist_out'])} \\\\")
    key = {"Abstract Expressionism": "Ae", "Cubism": "Cu", "Impressionism": "Im", "Surrealism": "Su", "all": "All"}[mv]
    m2(f"As{key}Full", f1(g.loc[mv, "full_pool"])); m2(f"As{key}Loao", f1(g.loc[mv, "leave_one_artist_out"]))
    m2(f"As{key}Drop", f1(g.loc[mv, "full_pool"] - g.loc[mv, "leave_one_artist_out"]))
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_ashton.tex").write_text("\n".join(lines) + "\n")

# factorial near-duplicate audit (C6)
FD = pd.read_csv(RV / "factorial_dup_summary.csv")
for d, t in DS.items():
    for des in ["emf", "mf"]:
        z = FD[(FD.dataset == d) & (FD.design == des)]
        m2(f"{t}FdPct{des.title()}", f1(100 * z.flagged.sum() / z.n_test.sum(), 2))

# exposure-matched condition-level figure (main) from per-image predictions
def cond_means(dataset, design):
    cm = {"emf": ["emf_AG", "emf_A", "emf_G", "emf_none"], "mf": ["mf_iid", "mf_genre", "mf_artist", "mf_joint"]}[design]
    out = {}
    for e in ENC:
        Pq = pd.read_parquet(A / "results" / "preds" / dataset / f"{e}.parquet")
        Pq = Pq[(Pq.evaluator == "knn5") & Pq.protocol.isin(cm)]
        b = Pq.assign(c=Pq.y == Pq.pred).groupby(["protocol", "seed", "style"]).c.mean().groupby(["protocol", "seed"]).mean()
        out[e] = {p: (100 * b.loc[p].mean(), 100 * b.loc[p].std(ddof=1)) for p in cm}
    return cm, out


fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
for ax, d in zip(axes, DS):
    cm, out = cond_means(d, "emf")
    ends = []
    for e in ENC:
        y = [out[e][p][0] for p in cm]
        sd = [out[e][p][1] for p in cm]
        ax.errorbar(range(4), y, yerr=sd, color=FAMCOL[FAM[e]], lw=1.4, marker=FAMMARK[FAM[e]], ms=3.5, capsize=1.5, elinewidth=0.8)
        ends.append(y[-1])
    label_ends(ax, None, ends, [NICE[e] for e in ENC], [FAMCOL[FAM[e]] for e in ENC], 3, (ax.get_ylim()[1] - ax.get_ylim()[0]) / 21)
    ax.set_xticks(range(4), ["A+G\nseen", "A seen\nG unseen", "G seen\nA unseen", "neither\nseen"], fontsize=6.3)
    ax.set_xlim(-0.25, 4.75)
    ax.set_title(DSN[d], fontsize=8.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylabel("balanced accuracy (%), kNN-5")
for f_, c in FAMCOL.items():
    axes[0].plot([], [], color=c, lw=1.6, marker=FAMMARK[f_], ms=4, label=FAMNAME[f_])
fig.legend(fontsize=6.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(P / "figures" / "fig_emf.pdf")
plt.close(fig)

# replication table fixed (A1, A2): images after cleaning (main/variants) or of the subset; EMF columns where available
lines = [r"\begin{tabular}{l r cccc cc cc}", r"\toprule",
         r" & & & & & & \multicolumn{2}{c}{Matched} & \multicolumn{2}{c}{Exposure-matched} \\",
         r"\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"Data & Images & IID & Artist-D & Genre-D & Joint & $\Delta_{\mathrm{G}\mid\mathrm{A}^{+}}$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{+}}$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ \\", r"\midrule"]
VAR["mp_novel"] = "MultitaskPainting100k without a WikiArt counterpart"
for v, t in vk.items():
    nimg = R1["NVimages"] if v == "mp_novel" else R1[t + "vN"]
    es = FSUM.get(f"{v}|emf|knn5")
    emf_cells = [f1(100 * es["genre_Aplus_mean"]), f1(100 * es["genre_Aminus_mean"])] if es else ["--", "--"]
    lines.append(f"{VAR[v]} & {nimg} & " + " & ".join(R1[f"{t}v{k}"] for k in ["Iid", "Art", "Gen", "Joint"])
                 + f" & {R1[t+'vGs']} & {R1[t+'vGu']} & " + " & ".join(emf_cells) + " \\\\")
    if v in ("wikiart_keepconf", "multitask_painting_keepconf"):
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_replication.tex").write_text("\n".join(lines) + "\n")

# misc round-2 macros
s5 = pd.read_csv(A / "splits" / "wikiart" / "split_summary.csv"); s6 = pd.read_csv(A / "splits" / "multitask_painting" / "split_summary.csv")
for t, ss in [("WA", s5), ("MP", s6)]:
    g = ss.groupby("protocol")[["n_train_total", "n_test", "n_test_artists"]].mean()
    m2(f"{t}GenTestPct", f1(100 * g.loc["genre_disjoint", "n_test"] / (g.loc["genre_disjoint", "n_test"] + g.loc["genre_disjoint", "n_train_total"]), 0))
    m2(f"{t}GenTrainRed", f1(100 * (1 - g.loc["genre_disjoint", "n_train_total"] / g.loc["iid", "n_train_total"]), 0))
    m2(f"{t}JointTrainRed", f1(100 * (1 - g.loc["joint", "n_train_total"] / g.loc["iid", "n_train_total"]), 0))
    m2(f"{t}ArtToJoint", f1(float(M[f"{t}meanArt"]) - float(M[f"{t}meanJoint"])))
with open(P / "numbers.tex", "a") as fh:
    fh.write("% ---- revision round 2\n")
    for k in sorted(R2):
        fh.write(f"\\newcommand{{\\{k}}}{{{R2[k]}}}\n")
M.update(R2)
json.dump(M, open(P / "numbers.json", "w"), indent=0)
print(f"revision round 2: {len(R2)} macros")

# ======================================================================== revision round 3 assets
R3 = {}
RV3 = F / "rev3"


def m3(name, val):
    assert name.isalpha(), name
    R3[name] = val


# add-on share sensitivity (R1-C2): base set subsampled to a fraction f, add-ons and test set unchanged
FAO = json.load(open(F / "factorial_addon" / "summary.json"))
AOWORD = {100: "Full", 50: "Half", 25: "Quarter", 10: "Tenth"}
lines = [r"\begin{tabular}{l c c cccc c ccc}", r"\toprule",
         r" & & & \multicolumn{5}{c}{$k$NN} & \multicolumn{3}{c}{nearest centroid} \\",
         r"\cmidrule(lr){4-8}\cmidrule(lr){9-11}",
         r"Dataset & Base kept & Add-on share (\%) & $\Delta_{\mathrm{A}}$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{+}}$ & "
         r"$\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ & $I$ & $I>0$ & $\Delta_{\mathrm{G}\mid\mathrm{A}^{+}}$ & "
         r"$\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ & $I$ \\", r"\midrule"]
for d, t_ in DS.items():
    rep = pd.read_csv(A / "reports" / f"{d}_addon_share_report.csv").groupby("base_fraction").addon_share.mean()
    for pc in [100, 50, 25, 10]:
        key = d if pc == 100 else f"{d}_ao{pc}"
        src = FSUM if pc == 100 else FAO
        k_, n_ = src[f"{key}|emf|knn5"], src[f"{key}|emf|ncc"]
        assert k_["n_encoders"] == len(ENC) and n_["n_encoders"] == len(ENC)
        sh = 100 * rep.loc[pc / 100]
        lines.append((DSN[d] if pc == 100 else "") + f" & {pc}\\% & {f1(sh)} & " + " & ".join(
            [f1(100 * k_[x]) for x in ["artist_mean", "genre_Aplus_mean", "genre_Aminus_mean", "interaction_mean"]]
            + [f"{k_['interaction_pos_bca']}/{k_['n_encoders']}"]
            + [f1(100 * n_[x]) for x in ["genre_Aplus_mean", "genre_Aminus_mean", "interaction_mean"]]) + r" \\")
        w = AOWORD[pc]
        m3(f"{t_}ao{w}Share", f1(sh, 0))
        m3(f"{t_}ao{w}Gs", f1(100 * k_["genre_Aplus_mean"])); m3(f"{t_}ao{w}Gu", f1(100 * k_["genre_Aminus_mean"]))
        m3(f"{t_}ao{w}I", f1(100 * k_["interaction_mean"])); m3(f"{t_}ao{w}Ipos", str(k_["interaction_pos_bca"]))
        m3(f"{t_}ao{w}Gumax", f1(100 * k_["genre_Aminus_max_upper_bca"]))
        m3(f"{t_}ao{w}NccGs", f1(100 * n_["genre_Aplus_mean"])); m3(f"{t_}ao{w}NccGu", f1(100 * n_["genre_Aminus_mean"]))
        m3(f"{t_}ao{w}NccI", f1(100 * n_["interaction_mean"])); m3(f"{t_}ao{w}NccIpos", str(n_["interaction_pos_bca"]))
    if d == "wikiart":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_addon.tex").write_text("\n".join(lines) + "\n")

# eligibility of test artists for the exposure-matched design (R2-C2)
EL = pd.read_csv(RV3 / "emf_eligibility.csv")
ELC = [("n_images", "images in pool", 1, 1), ("n_styles", "styles", 1, 2), ("n_genres", "genres", 1, 1),
       ("heldout_genre_share", "share in held-out genres (\\%)", 100, 0), ("acc_iid", "IID accuracy (\\%)", 100, 1),
       ("acc_artist_disjoint", "Artist-D accuracy (\\%)", 100, 1), ("acc_joint", "Joint accuracy (\\%)", 100, 1)]
gm = EL.groupby(["dataset", "encoder", "seed", "eligible"])[[c for c, *_ in ELC]].mean()
gm["n_artists"] = EL.groupby(["dataset", "encoder", "seed", "eligible"]).size()
ge = gm.groupby(["dataset", "eligible"]).mean()
dif = (gm.xs(True, level="eligible") - gm.xs(False, level="eligible")).groupby(["dataset", "encoder"]).mean()
img = EL[EL.encoder == ENC[0]].groupby(["dataset", "seed", "eligible"]).n_images.sum().unstack()
imgshare = (img[True] / img.sum(1)).groupby("dataset").mean()
lines = [r"\begin{tabular}{l cc c cc c}", r"\toprule",
         r" & \multicolumn{3}{c}{WikiArt} & \multicolumn{3}{c}{MultitaskPainting100k} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Per artist (mean) & evaluated & not evaluated & diff.\ range & evaluated & not evaluated & diff.\ range \\", r"\midrule"]
lines.append("artists per seed & " + " & ".join(f"{ge.loc[(d, True), 'n_artists']:.0f} & {ge.loc[(d, False), 'n_artists']:.0f} & --" for d in DS) + r" \\")
for c, nm, sc, dd in ELC:
    cells = []
    for d in DS:
        rg = dif.loc[d][c] * sc
        acc = c.startswith("acc_")
        cells.append(f"{f1(sc * ge.loc[(d, True), c], dd)} & {f1(sc * ge.loc[(d, False), c], dd)} & "
                     + (f"{f1(rg.min())} to {f1(rg.max())}" if acc else "--"))
    lines.append(f"{nm} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_eligibility.tex").write_text("\n".join(lines) + "\n")
BR = pd.read_csv(RV3 / "emf_breadth_effects.csv")
br = BR.groupby(["dataset", "group"])[["artist", "genre_Aplus", "genre_Aminus", "interaction", "n_artists"]].mean()
for d, t_ in DS.items():
    m3(f"{t_}ElN", f1(ge.loc[(d, True), "n_artists"], 0)); m3(f"{t_}ElNot", f1(ge.loc[(d, False), "n_artists"], 0))
    m3(f"{t_}ElImgShare", f1(100 * imgshare.loc[d], 0))
    m3(f"{t_}ElPop", f1(ge.loc[(d, True), "n_artists"] + ge.loc[(d, False), "n_artists"], 0))
    m3(f"{t_}ElGenres", f1(ge.loc[(d, True), "n_genres"])); m3(f"{t_}ElNotGenres", f1(ge.loc[(d, False), "n_genres"]))
    for c, w in [("acc_iid", "Iid"), ("acc_artist_disjoint", "Art"), ("acc_joint", "Joint")]:
        m3(f"{t_}El{w}", f1(100 * ge.loc[(d, True), c])); m3(f"{t_}ElNot{w}", f1(100 * ge.loc[(d, False), c]))
        m3(f"{t_}ElDiff{w}", f1(100 * (ge.loc[(d, True), c] - ge.loc[(d, False), c])))
    for g_, w in [("few_genres", "Few"), ("many_genres", "Many")]:
        m3(f"{t_}Br{w}I", f1(100 * br.loc[(d, g_), "interaction"])); m3(f"{t_}Br{w}Gu", f1(100 * br.loc[(d, g_), "genre_Aminus"]))
        m3(f"{t_}Br{w}Gs", f1(100 * br.loc[(d, g_), "genre_Aplus"]))
    assert (BR[BR.dataset == d].groupby(["encoder", "group"]).interaction.mean() > 0).all()

# statistical sensitivity (R2-C4)
SS = pd.read_csv(RV3 / "stat_sensitivity.csv")
OV = pd.read_csv(RV3 / "seed_overlap.csv").set_index("dataset")
FQ = FA[(FA.design == "emf") & (FA.evaluator == "knn5")]
SSDS = [("wikiart", "WikiArt"), ("multitask_painting", "MultitaskPainting100k"), ("wikiart_ga", "WikiArt, genre-agnostic"),
        ("multitask_painting_ga", "MP100k, genre-agnostic"), ("mp_novel", "MP100k without WikiArt counterpart")]
lines = [r"\begin{tabular}{l c cc cccc c cc}", r"\toprule",
         r" & & \multicolumn{2}{c}{$I$ (points)} & \multicolumn{4}{c}{encoders with $I$ interval $>0$} & width of $I$ interval & \multicolumn{2}{c}{$\Delta_{\mathrm{G}\mid\mathrm{A}^{-}}$ interval $>0$} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}\cmidrule(lr){10-11}",
         r"Data & Jaccard & style-macro & artist-macro & hier.\ BCa & global & global Bonf. & artist-macro & seed / hier. / global & hier.\ BCa & global Bonf. \\", r"\midrule"]
for d, nm in SSDS:
    s_ = SS[SS.dataset == d]
    it, gu = s_[s_.effect == "interaction"], s_[s_.effect == "genre_Aminus"]
    fq = FQ[(FQ.dataset == d)]
    fi, fg = fq[fq.effect == "interaction"], fq[fq.effect == "genre_Aminus"]
    assert len(it) == len(ENC) == len(fi)
    w = [100 * (fi.seed_hi - fi.seed_lo).mean(), 100 * (fi.hier_hi - fi.hier_lo).mean(), 100 * (it.glob_hi - it.glob_lo).mean()]
    lines.append(f"{nm} & {OV.loc[d, 'mean_pairwise_jaccard']:.2f} & {f1(100 * it.style_macro.mean())} & {f1(100 * it.artist_macro.mean())} & "
                 f"{(fi.bca_lo > 0).sum()}/{len(fi)} & {(it.glob_lo > 0).sum()}/{len(it)} & {(it.glob_bonf_lo > 0).sum()}/{len(it)} & "
                 f"{(it.am_lo > 0).sum()}/{len(it)} & " + " / ".join(f1(x) for x in w)
                 + f" & {(fg.bca_lo > 0).sum()}/{len(fg)} & {(gu.glob_bonf_lo > 0).sum()}/{len(gu)}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables" / "tab_statsens.tex").write_text("\n".join(lines) + "\n")
for d, t_ in [("wikiart", "WA"), ("multitask_painting", "MP"), ("mp_novel", "NV")]:
    s_ = SS[SS.dataset == d]
    it, gu, gs = (s_[s_.effect == k] for k in ["interaction", "genre_Aminus", "genre_Aplus"])
    m3(f"{t_}SsIam", f1(100 * it.artist_macro.mean())); m3(f"{t_}SsIsm", f1(100 * it.style_macro.mean()))
    m3(f"{t_}SsGsam", f1(100 * gs.artist_macro.mean())); m3(f"{t_}SsGuam", f1(100 * gu.artist_macro.mean()))
    m3(f"{t_}SsIampos", str(int((it.am_lo > 0).sum()))); m3(f"{t_}SsIglobpos", str(int((it.glob_lo > 0).sum())))
    m3(f"{t_}SsIbonfpos", str(int((it.glob_bonf_lo > 0).sum()))); m3(f"{t_}SsGubonfpos", str(int((gu.glob_bonf_lo > 0).sum())))
    m3(f"{t_}SsJaccard", f"{OV.loc[d, 'mean_pairwise_jaccard']:.2f}")
    m3(f"{t_}SsSeedsPerArtist", f1(OV.loc[d, "mean_seeds_per_artist"]))
AS = pd.read_csv(RV3 / "artist_size_effects.csv")
asz = AS.groupby(["dataset", "tercile"]).interaction.mean()
for d, t_ in DS.items():
    m3(f"{t_}SzSmallI", f1(100 * asz.loc[(d, "small")])); m3(f"{t_}SzLargeI", f1(100 * asz.loc[(d, "large")]))
    assert asz.loc[(d, "small")] < asz.loc[(d, "mid")] < asz.loc[(d, "large")]
q = FQ[FQ.dataset.isin(["wikiart", "multitask_painting"])]
m3("SsDropShare", f1(100 * q.share_reps_style_dropped.min(), 0))
m3("SsSignAgree", str(int(((q.alt_nan_lo > 0) == (q.hier_lo > 0)).sum())))
assert set(q[(q.alt_nan_lo > 0) != (q.hier_lo > 0)].effect) <= {"genre_Aminus"}
m3("SsSignN", str(len(q)))
m3("SsAltShiftMax", f1(100 * (q.alt_nan_lo - q.hier_lo).abs().max()))
with open(P / "numbers.tex", "a") as fh:
    fh.write("% ---- revision round 3\n")
    for k in sorted(R3):
        fh.write(f"\\newcommand{{\\{k}}}{{{R3[k]}}}\n")
M.update(R3)
json.dump(M, open(P / "numbers.json", "w"), indent=0)
print(f"revision round 3: {len(R3)} macros")
