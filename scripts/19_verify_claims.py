"""Revision: assert the qualitative claims made in the manuscript text against the results (fails loudly)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
F = A / "final"
M = json.load(open(ROOT / "paper" / "numbers.json"))
ok, bad = [], []


def check(name, cond):
    (ok if cond else bad).append(name)


for d, t in [("wikiart", "WA"), ("multitask_painting", "MP")]:
    E = pd.read_csv(F / d / "encoder_level.csv", index_col=0)
    P = ["iid", "artist_disjoint", "genre_disjoint", "joint", "mf_iid", "mf_artist", "mf_genre", "mf_joint"]
    check(f"{t}: every encoder loses under every control",
          ((E.iid > E.artist_disjoint) & (E.iid > E.genre_disjoint) & (E.iid > E.joint)).all())
    check(f"{t}: SigLIP SO400M first, MAE last everywhere",
          set(E[P].idxmax()) == {"siglip_so400m"} and set(E[P].idxmin()) == {"mae_b16"})
    check(f"{t}: CSD largest artist gap", E.artist_gap.idxmax() == "csd_l14")
    vl = E.loc[["clip_b16", "clip_l14", "openclip_b16", "siglip_b16", "siglip2_l16", "siglip_so400m"]]
    check(f"{t}: CSD/ALADIN do not outperform best VL under controlled protocols",
          all(E.loc[e, p] < vl[p].max() for e in ["csd_l14", "aladin_vit"] for p in ["artist_disjoint", "genre_disjoint", "joint"]))
    exc = set(E.index[E.genre_disjoint < E.artist_disjoint])
    check(f"{t}: Genre-D<Artist-D exceptions", exc == ({"dinov2_b14", "dinov2_l14", "vit_b16_in21k"} if t == "WA" else set()))
    for k in ["mean", "Fone", "Acc", "Ks", "Lr", "Ncc", "Len", "Str"]:
        v = {o: float(M[f"{t}{k}{o}"]) for o in ["Iid", "Art", "Gen", "Joint", "MfArt", "MfJoint"]}
        check(f"{t}/{k}: IID highest, Joint lowest", v["Iid"] > max(v["Art"], v["Gen"]) and v["Joint"] < min(v["Art"], v["Gen"]))
        check(f"{t}/{k}: genres-seen vs neither <= 3.5", abs(v["MfArt"] - v["MfJoint"]) <= 3.5)
        if k in ("Lr", "Ncc") and t == "WA":
            check(f"WA/{k}: Artist-D above Genre-D", v["Art"] > v["Gen"])
        elif k not in ("Lr", "Ncc"):
            check(f"{t}/{k}: Artist-D below Genre-D", v["Art"] < v["Gen"])
    if t == "MP":
        check("MP/Lr,Ncc: Artist-D below Genre-D", all(float(M[f"MP{k}Art"]) < float(M[f"MP{k}Gen"]) for k in ["Lr", "Ncc"]))
FS = json.load(open(F / "factorial" / "summary.json"))
check("MF interaction hier>0 all 26", sum(FS[f"{d}|mf|knn5"]["interaction_pos_hier"] for d in ["wikiart", "multitask_painting"]) == 26)
for d in ["wikiart", "multitask_painting", "mp_novel"]:
    check(f"EMF interaction hier>0 all 13 ({d})", FS[f"{d}|emf|knn5"]["interaction_pos_hier"] == 13)
    check(f"EMF genre|A- upper<5 all 13 ({d})", FS[f"{d}|emf|knn5"]["genre_Aminus_upper_below_delta_hier"] == 13)
    check(f"EMF margin is about half of smallest genre|A+ ({d})", 0.05 <= FS[f"{d}|emf|knn5"]["genre_Aplus_mean"] * 0.6)
check("5-pt margin <= half of smallest average genre|A+ (rounded)", min(
    FS[f"{d}|{des}|knn5"]["genre_Aplus_mean"] for d in ["wikiart", "multitask_painting"] for des in ["mf", "emf"]) / 2 >= 0.05)
for d in ["wikiart", "multitask_painting"]:
    x = pd.read_csv(A / "results" / "neighbor_intervention" / f"{d}.csv")
    x5 = x[x.k == 5]
    g = x5.groupby(["encoder", "condition"]).bacc.mean().unstack()
    check(f"{d}: random control drop <= 1.2 pts", (100 * (g["none"] - g["control_random"])).max() <= 1.2)
    check(f"{d}: same-artist drop >= 9.5 pts", (100 * (g["none"] - g["same_artist"])).min() >= 9.5)
    gt = int(((g["none"] - g["same_artist"]) > (g["none"] - g["control_nearest"])).sum())
    check(f"{d}: same-artist > nearest control for most encoders ({gt})", gt >= 12)
    check(f"{d}: same-style-only removal similar to any-style removal",
          (100 * ((g["none"] - g["same_artist_style"]) - (g["none"] - g["same_artist"])).abs()).max() <= 4.0)
    gk = x.groupby(["k", "condition"]).bacc.mean().unstack()
    diff = (gk["control_nearest"] - gk["same_artist"])          # same-artist cost minus rank-matched cost
    check(f"{d}: same-artist vs rank-matched difference shrinks from k=5 to k=20", diff.loc[20] < diff.loc[5])
for base, vs in [("wikiart", ["wikiart_phash8", "wikiart_keepconf"]),
                 ("multitask_painting", ["multitask_painting_phash8", "multitask_painting_keepconf"])]:
    b = pd.read_csv(F / base / "encoder_level.csv", index_col=0)
    for v in vs:
        e = pd.read_csv(F / v / "encoder_level.csv", index_col=0)
        diffs = [abs(e[p].mean() - b[p].mean()) for p in ["iid", "artist_disjoint", "genre_disjoint", "joint"]]
        check(f"{v}: protocol means differ < 1 pt", max(diffs) * 100 < 1.0)
        check(f"{v}: genre|A+ differs <= 2 pts", abs((e.mf_iid - e.mf_genre).mean() - (b.mf_iid - b.mf_genre).mean()) * 100 <= 2.0)
        check(f"{v}: genre|A- differs < 0.5 pt", abs((e.mf_artist - e.mf_joint).mean() - (b.mf_artist - b.mf_joint).mean()) * 100 < 0.5)
ps = pd.read_csv(F / "factorial" / "perstyle_effects.csv")
ps = ps[(ps.design == "emf") & (ps.evaluator == "knn5")].groupby(["dataset", "style", "effect"]).value.mean().unstack()
check("WA per-style interaction all positive", (ps.loc["wikiart"].interaction > 0).all())
# ---------------- revision round 2 claims
import re  # noqa: E402
RV = A / "results" / "rev2"
for d, t in [("wikiart", "WA"), ("multitask_painting", "MP")]:
    E = pd.read_csv(F / d / "encoder_level.csv", index_col=0)
    for p in ["artist_disjoint", "genre_disjoint", "joint"]:
        top2 = E.loc[["siglip_so400m", "siglip2_l16"], p].min()
        check(f"{t}/{p}: style-specific encoders below SigLIP SO400M and SigLIP2", E.loc[["csd_l14", "aladin_vit"], p].max() < top2)
    small = ["clip_b16", "openclip_b16", "siglip_b16", "clip_l14"]
    n_ex = int(sum((E.loc["csd_l14", p] > E.loc[sm, p]) for sm in small for p in ["artist_disjoint", "genre_disjoint", "joint"]))
    check(f"{t}: CSD exceeds several smaller VL encoders ({n_ex} cases)", n_ex >= 2)
FS2 = json.load(open(F / "factorial" / "summary.json"))
for d in ["wikiart", "multitask_painting", "wikiart_ga", "multitask_painting_ga", "mp_novel"]:
    check(f"EMF interaction BCa>0 for all 13 ({d})", FS2[f"{d}|emf|knn5"]["interaction_pos_bca"] == 13)
    check(f"EMF genre|A+ > genre|A- ({d})", FS2[f"{d}|emf|knn5"]["genre_Aplus_mean"] > 3 * FS2[f"{d}|emf|knn5"]["genre_Aminus_mean"])
for d in ["wikiart", "multitask_painting"]:
    check(f"MF interaction smaller on genre-agnostic partition ({d})",
          FS2[f"{d}_ga|mf|knn5"]["interaction_mean"] < FS2[f"{d}|mf|knn5"]["interaction_mean"])
    check(f"MF interaction still positive for >=12 encoders on genre-agnostic partition ({d})",
          FS2[f"{d}_ga|mf|knn5"]["interaction_pos_bca"] >= 12)
    nc = FS2[f"{d}|emf|ncc"]
    check(f"NCC EMF: no appreciable artist effect or interaction ({d})", abs(nc["artist_mean"]) < 0.035 and abs(nc["interaction_mean"]) < 0.01)
    if f"{d}|emf|logreg" in FS2 and FS2[f"{d}|emf|logreg"]["n_encoders"] == 13:
        lr = FS2[f"{d}|emf|logreg"]; kn = FS2[f"{d}|emf|knn5"]
        check(f"LR EMF: same pattern, smaller effects ({d})", lr["genre_Aplus_mean"] > lr["genre_Aminus_mean"]
              and lr["genre_Aplus_mean"] < kn["genre_Aplus_mean"])
ZS = pd.read_csv(RV / "zero_shot.csv")
for d, t in [("wikiart", "WA"), ("multitask_painting", "MP")]:
    z = ZS[ZS.dataset == d].groupby("protocol").bacc.mean() * 100
    check(f"{t}: zero-shot almost equal across test sets (range {z.max()-z.min():.1f})", z.max() - z.min() <= 4.0)
check("WA: kNN Artist-D barely above zero-shot (VL encoders)", 0 <= float(M["WAZsKnnArt"]) - float(M["WAZsArt"]) <= 3.0)
DC = pd.read_csv(RV / "decomposition.csv")
for d in ["wikiart", "multitask_painting"]:
    m = DC[DC.dataset == d].mean(numeric_only=True)
    comp = [m.A_artistD - m.B_jointpart_full, m.B_jointpart_full - m.C_jointpart_sizematched, m.C_jointpart_sizematched - m.D_joint]
    check(f"{d}: decomposition sums to total", abs(sum(comp) - (m.A_artistD - m.D_joint)) < 1e-9)
    check(f"{d}: test-set/partition component largest", comp[0] > comp[1] and comp[0] > comp[2])
    sz = pd.read_csv(RV / "size_control.csv"); sz = sz[sz.dataset == d].mean(numeric_only=True)
    check(f"{d}: training size explains little of IID->Joint", (sz.iid - sz.iid_at_joint_size) < 0.25 * (sz.iid - sz.joint))
SB = pd.read_csv(RV / "selection_bias.csv")
for d in ["wikiart", "multitask_painting"]:
    j = SB[(SB.dataset == d) & (SB.partition == "joint")].mean(numeric_only=True)
    ga = SB[(SB.dataset == d) & (SB.partition == "genre_agnostic")].mean(numeric_only=True)
    check(f"{d}: Joint partition genre-dependent", j.test_artists_heldout_share - j.train_artists_heldout_share > 0.3)
    check(f"{d}: genre-agnostic partition balanced", abs(ga.test_artists_heldout_share - ga.train_artists_heldout_share) < 0.03)
AS = pd.read_csv(RV / "ashton_replication.csv")
g = AS[AS.movement != "all"].groupby(["movement", "protocol"]).acc.mean().unstack()
drop = g.full_pool - g.leave_one_artist_out
check("Ashton: Surrealism largest drop", drop.idxmax() == "Surrealism")
check("Ashton: Impressionism smallest drop", drop.idxmin() == "Impressionism")
check("Ashton: Cubism drops markedly (>10 pts)", 100 * drop["Cubism"] > 10)
hl = open(ROOT / "paper" / "highlights.txt", encoding="utf8").read()
check("highlights <= 85 chars", all(len(l) <= 85 for l in hl.splitlines()))
check("highlight intervention range matches", f"{M['WAIntvDroplo']}–{M['WAIntvDrophi']}" in hl)
check("highlight Kendall tau matches", f"τ = {M['WAtauJoint']}" in hl)
tex = open(ROOT / "paper" / "main.tex").read()
ab = tex[tex.index("\\begin{abstract}") + 16: tex.index("\\end{abstract}")]
ab = re.sub(r"\\([A-Za-z]+)(\{\})?", lambda mm: str(M.get(mm.group(1), "")), ab)
check(f"abstract <= 250 words ({len(ab.split())})", len(ab.split()) <= 250)
alltex = tex + "".join(open(f).read() for f in sorted((ROOT / "paper" / "tex").glob("*.tex")))
alltex = "\n".join(l.split("%")[0] if not l.strip().startswith("%") else "" for l in alltex.splitlines())
check("no 'mediated' wording", "mediated" not in alltex.replace("mediation", ""))
check("no singular 'the author' wording (three authors)", re.search(r"\bthe author\b", alltex.lower()) is None)
for k_, v_ in FS2.items():
    check(f"summary {k_}: at most 13 encoders", v_["n_encoders"] <= 13)
# ---- revision round 3 claims
FAO = json.load(open(F / "factorial_addon" / "summary.json"))
FM = json.load(open(F / "factorial" / "summary.json"))
for d in ["wikiart", "multitask_painting"]:
    seq = [FM[f"{d}|emf|knn5"]] + [FAO[f"{d}_ao{p}|emf|knn5"] for p in (50, 25, 10)]
    nseq = [FM[f"{d}|emf|ncc"]] + [FAO[f"{d}_ao{p}|emf|ncc"] for p in (50, 25, 10)]
    check(f"{d}: add-on kNN interaction positive for all encoders at every share", all(s["interaction_pos_bca"] == 13 for s in seq))
    check(f"{d}: add-on kNN genre|A- stays < 3 pts", all(s["genre_Aminus_mean"] < 0.03 for s in seq))
    check(f"{d}: add-on kNN genre|A- grows only slightly (<2 pts)", seq[-1]["genre_Aminus_mean"] - seq[0]["genre_Aminus_mean"] < 0.02)
    check(f"{d}: add-on kNN genre|A+ stable (within 1 pt)", max(s["genre_Aplus_mean"] for s in seq) - min(s["genre_Aplus_mean"] for s in seq) < 0.01)
    check(f"{d}: NCC genre effects grow with share", nseq[-1]["genre_Aminus_mean"] > 3 * nseq[0]["genre_Aminus_mean"] and nseq[-1]["genre_Aplus_mean"] > 3 * nseq[0]["genre_Aplus_mean"])
    check(f"{d}: NCC no interaction at any share", all(abs(s["interaction_mean"]) < 0.01 for s in nseq))
    rep = pd.read_csv(A / "reports" / f"{d}_addon_share_report.csv").groupby("base_fraction").addon_share.mean()
    check(f"{d}: largest add-on share near half (40-50%)", 0.40 < rep.loc[0.1] < 0.5)
EL = pd.read_csv(F / "rev3" / "emf_eligibility.csv")
gm = EL.groupby(["dataset", "encoder", "seed", "eligible"])[["n_genres", "n_images", "acc_iid", "acc_joint"]].mean().groupby(["dataset", "eligible"]).mean()
for d in ["wikiart", "multitask_painting"]:
    check(f"{d}: evaluated artists have more genres and paintings",
          gm.loc[(d, True), "n_genres"] > gm.loc[(d, False), "n_genres"] and gm.loc[(d, True), "n_images"] > gm.loc[(d, False), "n_images"])
check("WA: IID accuracy of evaluated vs not similar (<1 pt)", abs(gm.loc[("wikiart", True), "acc_iid"] - gm.loc[("wikiart", False), "acc_iid"]) < 0.01)
check("MP: IID accuracy of evaluated vs not similar (<4 pts)", abs(gm.loc[("multitask_painting", True), "acc_iid"] - gm.loc[("multitask_painting", False), "acc_iid"]) < 0.04)
check("WA: Joint accuracy similar (<1.5 pt)", abs(gm.loc[("wikiart", True), "acc_joint"] - gm.loc[("wikiart", False), "acc_joint"]) < 0.015)
check("MP: Joint accuracy higher for evaluated (>5 pts)", gm.loc[("multitask_painting", True), "acc_joint"] - gm.loc[("multitask_painting", False), "acc_joint"] > 0.05)
BR = pd.read_csv(F / "rev3" / "emf_breadth_effects.csv")
check("breadth: interaction positive in both groups for every encoder", (BR.groupby(["dataset", "encoder", "group"]).interaction.mean() > 0).all())
SS = pd.read_csv(F / "rev3" / "stat_sensitivity.csv")
it = SS[SS.effect == "interaction"]
check("statsens: global, Bonferroni and artist-macro intervals > 0 for all", (it.glob_lo > 0).all() and (it.glob_bonf_lo > 0).all() and (it.am_lo > 0).all())
check("statsens: artist-macro interaction smaller than style-macro (all datasets)",
      (it.groupby("dataset").artist_macro.mean() < it.groupby("dataset").style_macro.mean()).all())
ASZ = pd.read_csv(F / "rev3" / "artist_size_effects.csv").groupby(["dataset", "tercile"]).interaction.mean()
for d in ["wikiart", "multitask_painting"]:
    check(f"{d}: per-artist interaction increases with artist size", ASZ.loc[(d, "small")] < ASZ.loc[(d, "mid")] < ASZ.loc[(d, "large")])
check("moayeri cited as ICLR 2025", "ICLR" in open(ROOT / "paper" / "refs.bib").read().split("moayeri2024rethinking")[1][:600])
print(f"{len(ok)} claims verified")
for b_ in bad:
    print("FAILED:", b_)
sys.exit(1 if bad else 0)
