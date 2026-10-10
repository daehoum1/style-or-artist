"""Split protocols: iid, artist_disjoint, genre_disjoint, joint (artist + genre disjoint).

Output of every protocol: a table with columns image_id, style, artist, genre, part in {train, val, test}.
Images that are not used by the split are absent. Validation is always carved out of the train side and
mirrors the protocol's shift: iid -> random; artist -> held-out train artists; genre -> held-out train genre;
joint -> held-out train artists x held-out train genre.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

PROTOCOLS = ["iid", "artist_disjoint", "genre_disjoint", "joint"]
NEEDS_ARTIST_DISJOINT = {"artist_disjoint", "joint"}
NEEDS_GENRE_DISJOINT = {"genre_disjoint", "joint"}


# ----------------------------------------------------------------------------------------------- pool
def controlled_pool(clean: pd.DataFrame, sup: dict, need_genre: bool = True):
    """Rows with all required labels, restricted to the core style set. Returns (pool, style_table)."""
    req = ["style", "artist"] + (["genre"] if need_genre else [])
    pool = clean.dropna(subset=req)[["image_id"] + req + (["genre"] if not need_genre else [])].copy()
    if not need_genre:
        pool["genre"] = pd.NA
    g = pool.groupby("style")
    t = pd.DataFrame({"n_images": g.size(), "n_artists": g["artist"].nunique()})
    reasons = {}
    for s, r in t.iterrows():
        why = []
        if r.n_images < sup["min_images_per_style"]:
            why.append(f"n_images={r.n_images}<{sup['min_images_per_style']}")
        if r.n_artists < sup["min_artists_per_style"]:
            why.append(f"n_artists={r.n_artists}<{sup['min_artists_per_style']}")
        reasons[s] = why
    if need_genre:
        t["top_genre_share"] = g["genre"].agg(lambda x: x.value_counts(normalize=True).iloc[0]).round(3)
        for s, r in t.iterrows():
            if r.top_genre_share > sup["max_top_genre_share"]:
                reasons[s].append(f"top_genre_share={r.top_genre_share}>{sup['max_top_genre_share']}")
    t["excluded_reason"] = [";".join(reasons[s]) for s in t.index]
    core = t.index[t.excluded_reason == ""]
    return pool[pool["style"].isin(core)].reset_index(drop=True), t


# ------------------------------------------------------------------------------------------- helpers
def _style_support(df: pd.DataFrame, sup: dict, artist_rules: bool) -> dict:
    """style -> list of violated rules for a finished split (train side = train+val)."""
    tr = df[df.part != "test"]
    te = df[df.part == "test"]
    bad = {}
    for s in sorted(df["style"].unique()):
        a, b = tr[tr["style"] == s], te[te["style"] == s]
        why = []
        if len(a) < sup["min_train_per_style"]:
            why.append(f"train={len(a)}<{sup['min_train_per_style']}")
        if len(b) < sup["min_test_per_style"]:
            why.append(f"test={len(b)}<{sup['min_test_per_style']}")
        if artist_rules:
            if a["artist"].nunique() < sup["min_train_artists_per_style"]:
                why.append(f"train_artists={a['artist'].nunique()}<{sup['min_train_artists_per_style']}")
            if b["artist"].nunique() < sup["min_test_artists_per_style"]:
                why.append(f"test_artists={b['artist'].nunique()}<{sup['min_test_artists_per_style']}")
        if why:
            bad[s] = why
    return bad


def _stratified_holdout(df: pd.DataFrame, frac: float, rng, by="style") -> np.ndarray:
    """Boolean mask selecting ~frac of each stratum (at least 1 if stratum has >= 2)."""
    m = np.zeros(len(df), bool)
    for _, idx in df.groupby(by).indices.items():
        k = int(round(frac * len(idx)))
        if len(idx) >= 2:
            k = min(max(k, 1), len(idx) - 1)
        else:
            k = 0
        m[rng.choice(idx, size=k, replace=False)] = True
    return m


# --------------------------------------------------------------------------------- artist optimizer
def optimize_artists(pool: pd.DataFrame, test_genre_mask: np.ndarray | None, cfg: dict, sup: dict, rng,
                     target: float):
    """Binary assignment of artists to the test side.

    test_genre_mask: per row, True if the image's genre is a held-out genre (joint protocol), None for the
    artist-only protocol (every image of a test artist is test, every image of a train artist is train).
    Maximizes   retained - w_v * violations - w_f * mean_s (test_frac_s - target)^2   by randomized greedy
    flips with restarts. Purely metadata-driven.
    """
    artists, a_idx = np.unique(pool["artist"].to_numpy(), return_inverse=True)
    styles, s_idx = np.unique(pool["style"].to_numpy(), return_inverse=True)
    A, S = len(artists), len(styles)
    if test_genre_mask is None:
        te_ok = np.ones(len(pool), bool)
        tr_ok = np.ones(len(pool), bool)
    else:
        te_ok, tr_ok = test_genre_mask, ~test_genre_mask
    Mte = np.zeros((A, S))
    Mtr = np.zeros((A, S))
    np.add.at(Mte, (a_idx[te_ok], s_idx[te_ok]), 1)
    np.add.at(Mtr, (a_idx[tr_ok], s_idx[tr_ok]), 1)
    Bte, Btr = (Mte > 0).astype(float), (Mtr > 0).astype(float)
    N = len(pool)
    wv, wf = cfg["w_violation"], cfg["w_fraction"]
    mt, mtr = sup["min_test_per_style"], sup["min_train_per_style"]
    mat, matr = sup["min_test_artists_per_style"], sup["min_train_artists_per_style"]

    def score(te, tr, tea, tra):
        viol = (np.maximum(0, mt - te) / mt + np.maximum(0, mtr - tr) / mtr
                + np.maximum(0, mat - tea) / mat + np.maximum(0, matr - tra) / matr).sum()
        tot = te + tr
        frac = np.where(tot > 0, te / np.maximum(tot, 1), 0)
        return tot.sum() / N - wv * viol - wf * ((frac - target) ** 2).mean()

    best = None
    for _ in range(cfg["restarts"]):
        x = rng.random(A) < target
        te, tr = x @ Mte, (~x) @ Mtr
        tea, tra = x @ Bte, (~x) @ Btr
        cur = score(te, tr, tea, tra)
        for _sweep in range(cfg["sweeps"]):
            improved = False
            for a in rng.permutation(A):
                sgn = -1.0 if x[a] else 1.0  # moving to test (+) or back to train (-)
                te2, tr2 = te + sgn * Mte[a], tr - sgn * Mtr[a]
                tea2, tra2 = tea + sgn * Bte[a], tra - sgn * Btr[a]
                new = score(te2, tr2, tea2, tra2)
                if new > cur + 1e-12:
                    x[a] = not x[a]
                    te, tr, tea, tra, cur = te2, tr2, tea2, tra2, new
                    improved = True
            if not improved:
                break
        if best is None or cur > best[0]:
            best = (cur, x.copy())
    return set(artists[best[1]]), float(best[0])


# --------------------------------------------------------------------------------- genre candidates
def genre_holdout_candidates(pool: pd.DataFrame, sup: dict, gcfg: dict):
    """Feasible held-out genre sets. Exhaustive for <= max_exhaustive genres, otherwise a seeded random sample
    of subsets. If fewer than min_candidates subsets are feasible for every core style, the style with the
    lowest feasibility rate is removed from the core set (metadata only) until enough remain.
    Returns (candidates, removed_styles: dict style -> reason)."""
    genres = sorted(pool["genre"].unique())
    ct = pd.crosstab(pool["style"], pool["genre"])[genres]
    M = ct.to_numpy().astype(float)                      # styles x genres
    N = M.sum()
    lo, hi = gcfg["test_share_range"]
    sl = gcfg["slack"]
    G = len(genres)
    if G <= gcfg.get("max_exhaustive", 12):
        masks = np.array([[(i >> j) & 1 for j in range(G)] for i in range(1, 2 ** G - 1)], bool)
    else:
        r = np.random.default_rng(gcfg.get("sample_seed", 0))
        ps = r.uniform(0.1, 0.5, size=gcfg.get("n_samples", 200000))
        masks = r.random((len(ps), G)) < ps[:, None]
        masks = np.unique(masks[(masks.any(1)) & (~masks.all(1))], axis=0)
    te = masks.astype(float) @ M.T                       # subsets x styles
    tr = M.sum(1)[None, :] - te
    share_ok = (lo <= te.sum(1) / N) & (te.sum(1) / N <= hi)
    F = (te >= sl * sup["min_test_per_style"]) & (tr >= sl * sup["min_train_per_style"]) & share_ok[:, None]
    styles = list(ct.index)
    alive = np.ones(len(styles), bool)
    removed = {}
    need = gcfg.get("min_candidates", 10)
    while F[:, alive].all(1).sum() < need and alive.sum() > 2:
        rate = np.where(alive, F.mean(0), np.inf)
        j = int(np.argmin(rate))
        alive[j] = False
        removed[styles[j]] = f"genre-holdout infeasible (feasible in {F[:, j].mean():.4f} of candidate sets)"
    ok = F[:, alive].all(1)
    cands = [tuple(g for g, m in zip(genres, row) if m) for row in masks[ok]]
    return cands, removed


# ----------------------------------------------------------------------------------------- protocols
def _finish(df: pd.DataFrame, test_mask, keep_mask=None) -> pd.DataFrame:
    df = df.copy()
    df["part"] = np.where(test_mask, "test", "train")
    if keep_mask is not None:
        df = df[keep_mask]
    return df.reset_index(drop=True)


def split_iid(pool, cfg, rng):
    return _finish(pool, _stratified_holdout(pool, cfg["test_fraction"], rng))


def split_artist(pool, cfg, rng):
    A_te, sc = optimize_artists(pool, None, cfg["optimizer"], cfg["support"], rng, cfg["test_fraction"])
    df = _finish(pool, pool["artist"].isin(A_te).to_numpy())
    df.attrs["test_artists"] = sorted(A_te)
    df.attrs["optimizer_score"] = sc
    return df


def split_genre(pool, G_te, rng):
    df = _finish(pool, pool["genre"].isin(G_te).to_numpy())
    df.attrs["test_genres"] = list(G_te)
    return df


def split_joint(pool, cfg, G_te, rng):
    gmask = pool["genre"].isin(G_te).to_numpy()
    A_te, sc = optimize_artists(pool, gmask, cfg["optimizer"], cfg["support"], rng, cfg["test_fraction"])
    amask = pool["artist"].isin(A_te).to_numpy()
    keep = (amask & gmask) | (~amask & ~gmask)  # cross cells (test artist x train genre etc.) are discarded
    df = _finish(pool, amask & gmask, keep)
    df.attrs.update(test_genres=list(G_te), test_artists=sorted(A_te), optimizer_score=sc,
                    n_discarded_cross_cells=int((~keep).sum()))
    return df


def add_validation(df: pd.DataFrame, protocol: str, cfg: dict, rng) -> pd.DataFrame:
    """Carve validation from the train side, mirroring the protocol's shift. Rows of the train side that
    would leak the validation shift into the fitting set are marked 'unused_for_selection' (they are still
    part of the final training set: final models are refit on train+val with the selected hyperparameters)."""
    df = df.copy()
    tr = np.flatnonzero(df.part.to_numpy() == "train")
    sub = df.iloc[tr]
    val = np.zeros(len(tr), bool)
    fit_excl = np.zeros(len(tr), bool)
    if protocol == "iid":
        val = _stratified_holdout(sub, cfg["val_fraction"], rng)
    if protocol in NEEDS_ARTIST_DISJOINT:
        arts = np.array(sorted(sub["artist"].unique()))
        va = set(rng.choice(arts, size=max(1, int(round(cfg["val_artist_fraction"] * len(arts)))),
                            replace=False))
        am = sub["artist"].isin(va).to_numpy()
    if protocol in NEEDS_GENRE_DISJOINT:
        gens = np.array(sorted(sub["genre"].unique()))
        vg = {rng.choice(gens)}
        gm = sub["genre"].isin(vg).to_numpy()
    if protocol == "artist_disjoint":
        val = am
    elif protocol == "genre_disjoint":
        val = gm
    elif protocol == "joint":
        val = am & gm
        fit_excl = am ^ gm  # val-artist x fit-genre and fit-artist x val-genre: excluded from fitting
    part = df.part.to_numpy().copy()
    p = part[tr]
    p[val] = "val"
    p[fit_excl] = "train_selection_excluded"
    part[tr] = p
    df["part"] = part
    if protocol in NEEDS_ARTIST_DISJOINT:
        df.attrs["val_artists"] = sorted(va)
    if protocol in NEEDS_GENRE_DISJOINT:
        df.attrs["val_genres"] = sorted(vg)
    return df


def make_seed_splits(pool: pd.DataFrame, cfg: dict, seed: int, G_te, has_genre=True) -> dict:
    """All protocols for one seed, restricted to their common supported style set."""
    rng = np.random.default_rng(seed)
    protos = ["iid", "artist_disjoint"] + (["genre_disjoint", "joint"] if has_genre else [])
    raw = {}
    for p in protos:
        r = np.random.default_rng([seed, PROTOCOLS.index(p)])
        if p == "iid":
            raw[p] = split_iid(pool, cfg, r)
        elif p == "artist_disjoint":
            raw[p] = split_artist(pool, cfg, r)
        elif p == "genre_disjoint":
            raw[p] = split_genre(pool, G_te, r)
        else:
            raw[p] = split_joint(pool, cfg, G_te, r)
    dropped = {}
    for p, d in raw.items():
        for s, why in _style_support(d, cfg["support"], p in NEEDS_ARTIST_DISJOINT).items():
            dropped.setdefault(s, []).append(f"{p}: " + ", ".join(why))
    out = {}
    for p, d in raw.items():
        attrs = dict(d.attrs)
        d = d[~d["style"].isin(dropped)].reset_index(drop=True)
        # re-check artist support after dropping styles cannot change other styles' counts
        d = add_validation(d, p, cfg, np.random.default_rng([seed, PROTOCOLS.index(p), 99]))
        d.attrs.update(attrs)
        d.attrs.update(protocol=p, seed=seed, dropped_styles=dropped)
        out[p] = d
    if has_genre:
        styles = sorted(out["joint"]["style"].unique())
        mf = split_matched_factorial(pool, set(raw["joint"].attrs["test_artists"]), G_te, styles, cfg, seed)
        for p, d in mf.items():
            d.attrs["dropped_styles"] = dropped
        out.update(mf)
    return out


# --------------------------------------------------------------------------- matched factorial design
MF_PROTOCOLS = ["mf_iid", "mf_artist", "mf_genre", "mf_joint"]
MF_SEEN = {"mf_iid": (True, True), "mf_artist": (False, True), "mf_genre": (True, False),
           "mf_joint": (False, False)}  # (test artists seen in training?, test genres seen in training?)


def split_matched_factorial(pool: pd.DataFrame, A_te: set, G_te, styles, cfg: dict, seed: int) -> dict:
    """2x2 factorial on ONE fixed test set with size-matched training sets.

    Cells from the joint partition: TT = test artists x test genres, TA = test artists x train genres,
    AT = train artists x test genres, BB = train artists x train genres (the base).
    Test = a random half of TT (stratified by style). The four conditions differ only in which cells may
    contribute training images; every condition draws exactly |BB_s| training images per style s, so the
    training-set size per style is identical and only nuisance exposure changes:
        mf_joint : BB                      (artist unseen, genre unseen)
        mf_artist: BB + AT                 (artist unseen, genre seen)
        mf_genre : BB + TA                 (artist seen,   genre unseen)
        mf_iid   : BB + AT + TA + TT_rest  (artist seen,   genre seen)
    """
    rng = np.random.default_rng([seed, 4242])
    p = pool[pool["style"].isin(styles)].reset_index(drop=True)
    am = p["artist"].isin(A_te).to_numpy()
    gm = p["genre"].isin(G_te).to_numpy()
    cell = np.select([am & gm, am & ~gm, ~am & gm], ["TT", "TA", "AT"], "BB")
    tt = np.flatnonzero(cell == "TT")
    test_m = np.zeros(len(p), bool)
    test_m[tt[_stratified_holdout(p.iloc[tt], 0.5, rng)]] = True
    # styles whose half-cell test set is below support are dropped from the factorial design (recorded)
    st = p["style"].to_numpy()
    mf_dropped = {s: f"mf_test={int((test_m & (st == s)).sum())}<{cfg['support']['min_test_per_style']}"
                  for s in styles if (test_m & (st == s)).sum() < cfg["support"]["min_test_per_style"]}
    styles = [s for s in styles if s not in mf_dropped]
    test_m &= np.isin(st, styles)
    allowed = {"mf_joint": {"BB"}, "mf_artist": {"BB", "AT"}, "mf_genre": {"BB", "TA"},
               "mf_iid": {"BB", "AT", "TA", "TT"}}
    out = {}
    for c, cells in allowed.items():
        r = np.random.default_rng([seed, 4242, MF_PROTOCOLS.index(c)])
        cand = np.isin(cell, list(cells)) & ~test_m
        tr = []
        for s in styles:
            n_s = int(((cell == "BB") & (p["style"].to_numpy() == s)).sum())
            idx = np.flatnonzero(cand & (p["style"].to_numpy() == s))
            tr.append(r.choice(idx, size=n_s, replace=False))
        tr = np.concatenate(tr)
        d = p.iloc[np.concatenate([tr, np.flatnonzero(test_m)])].copy()
        d["part"] = ["train"] * len(tr) + ["test"] * int(test_m.sum())
        d["cell"] = cell[np.concatenate([tr, np.flatnonzero(test_m)])]
        d = d.reset_index(drop=True)
        side = np.flatnonzero(d.part.to_numpy() == "train")
        v = _stratified_holdout(d.iloc[side], cfg["val_fraction"], np.random.default_rng([seed, 4243]))
        part = d.part.to_numpy().copy()
        part[side[v]] = "val"
        d["part"] = part
        d.attrs.update(protocol=c, seed=seed, test_genres=list(G_te), seen=MF_SEEN[c], mf_dropped=mf_dropped,
                       cell_counts_train=d[d.part != "test"].cell.value_counts().to_dict())
        out[c] = d
    return out


# ------------------------------------------------------------- exposure-matched factorial (revision)
EMF_PROTOCOLS = ["emf_AG", "emf_A", "emf_G", "emf_none"]   # which of (test artists, test genres) are seen


def split_exposure_matched(pool: pd.DataFrame, A_te: set, G_te, styles, cfg: dict, seed: int) -> dict:
    """Exposure-matched 2x2 factorial on the cells of the Joint partition (TT, TA, AT, BB).

    For every (test artist a, style s) with >= 2 images in TT and >= 1 in TA: TT_{a,s} is split at random into
    a test half and a source half, and q_{a,s} = min(|source half|, |TA_{a,s}|). All conditions share one test
    set (the test halves) and one base training set B_s (|BB_s| - k_s images of BB_s, k_s = sum_a q_{a,s});
    they differ only in k_s add-on images per style:
        emf_AG   : q_{a,s} images of each test artist from its test-genre source half   (A seen, G seen)
        emf_A    : q_{a,s} images of each test artist from its training genres (TA)     (A seen, G unseen)
        emf_G    : k_s images of training artists in test genres (AT), matching the (style, genre) counts
                   of emf_AG where AT allows it                                          (A unseen, G seen)
        emf_none : the remaining k_s images of BB_s                                      (A unseen, G unseen)
    Hence emf_AG and emf_A have identical test-artist coverage and identical per-artist support, and
    every condition has the same per-style training size. Styles with |AT_s| < k_s or |BB_s| - k_s below the
    minimum training support are excluded (recorded)."""
    rng = np.random.default_rng([seed, 5151])
    sup = cfg["support"]
    p = pool[pool["style"].isin(styles)].reset_index(drop=True)
    am = p["artist"].isin(A_te).to_numpy()
    gm = p["genre"].isin(G_te).to_numpy()
    cell = np.select([am & gm, am & ~gm, ~am & gm], ["TT", "TA", "AT"], "BB")
    st, ar, ge = p["style"].to_numpy(), p["artist"].to_numpy(), p["genre"].to_numpy()
    test_idx, add = [], {c: [] for c in EMF_PROTOCOLS}
    base_idx, excluded, stats = [], {}, []
    for s in styles:
        ins = st == s
        tt_s, ta_s = np.flatnonzero(ins & (cell == "TT")), np.flatnonzero(ins & (cell == "TA"))
        at_s, bb_s = np.flatnonzero(ins & (cell == "AT")), np.flatnonzero(ins & (cell == "BB"))
        t_s, ag, a_ = [], [], []
        for a in sorted(set(ar[tt_s])):
            tta = rng.permutation(tt_s[ar[tt_s] == a])
            taa = ta_s[ar[ta_s] == a]
            if len(tta) < 2 or len(taa) == 0:
                continue
            half = (len(tta) + 1) // 2
            t_s += list(tta[:half])
            src = tta[half:]
            q = min(len(src), len(taa))
            ag += list(rng.choice(src, q, replace=False))
            a_ += list(rng.choice(taa, q, replace=False))
        k = len(ag)
        if len(t_s) < sup["min_test_per_style"]:
            excluded[s] = f"test={len(t_s)}<{sup['min_test_per_style']}"
            continue
        if len(at_s) < k or len(bb_s) - k < sup["min_train_per_style"]:
            excluded[s] = f"AT={len(at_s)}, BB={len(bb_s)}, k={k}"
            continue
        # emf_G: match (style, genre) counts of the emf_AG add-on where AT allows, fill the rest from AT_s
        g_add, used = [], np.zeros(len(p), bool)
        for g, c in pd.Series(ge[ag]).value_counts().items():
            cand = at_s[ge[at_s] == g]
            take = rng.choice(cand, min(c, len(cand)), replace=False)
            g_add += list(take)
            used[take] = True
        rest = at_s[~used[at_s]]
        g_add += list(rng.choice(rest, k - len(g_add), replace=False))
        bbp = rng.permutation(bb_s)
        base_idx += list(bbp[: len(bb_s) - k])
        add["emf_none"] += list(bbp[len(bb_s) - k:])
        add["emf_AG"] += ag
        add["emf_A"] += a_
        add["emf_G"] += g_add
        test_idx += t_s
        stats.append({"style": s, "k": k, "base": len(bb_s) - k, "test": len(t_s),
                      "test_artists": len(set(ar[t_s])),
                      "emf_G_genre_matched": float(np.mean([1.0])) if k == 0 else
                      float((pd.Series(ge[g_add]).value_counts().reindex(pd.Series(ge[ag]).value_counts().index)
                             .fillna(0).clip(upper=pd.Series(ge[ag]).value_counts()).sum()) / k)})
    test_idx, base_idx = np.array(test_idx, int), np.array(base_idx, int)
    out = {}
    vr = np.random.default_rng([seed, 5152])
    for c in EMF_PROTOCOLS:
        tr = np.concatenate([base_idx, np.array(add[c], int)])
        d = p.iloc[np.concatenate([tr, test_idx])].copy()
        d["part"] = ["train"] * len(tr) + ["test"] * len(test_idx)
        d["cell"] = cell[np.concatenate([tr, test_idx])]
        d["role"] = ["base"] * len(base_idx) + ["addon"] * len(add[c]) + ["test"] * len(test_idx)
        d = d.reset_index(drop=True)
        side = np.flatnonzero(d.part.to_numpy() == "train")
        v = _stratified_holdout(d.iloc[side], cfg["val_fraction"], vr)
        part = d.part.to_numpy().copy()
        part[side[v]] = "val"
        d["part"] = part
        d.attrs.update(protocol=c, seed=seed, test_genres=list(G_te), emf_excluded=excluded,
                       emf_style_stats=stats)
        out[c] = d
    return out


def validate_exposure_matched(sp: dict, A_te: set, G_te) -> dict:
    """Assertions specific to the exposure-matched design; raises AssertionError on violation."""
    r = {}
    T = {c: set(d[d.part == "test"].image_id) for c, d in sp.items()}
    assert len({frozenset(v) for v in T.values()}) == 1, "test sets differ"
    sizes = {c: d[d.part != "test"].groupby("style").size() for c, d in sp.items()}
    ref = sizes["emf_AG"]
    assert all((v.reindex(ref.index) == ref).all() for v in sizes.values()), "per-style training sizes differ"
    base = {c: set(d[d.role == "base"].image_id) for c, d in sp.items()}
    assert len({frozenset(v) for v in base.values()}) == 1, "base sets differ"
    allimg = {c: d.image_id.is_unique for c, d in sp.items()}
    assert all(allimg.values()), "duplicate image within a condition"
    te = sp["emf_AG"][sp["emf_AG"].part == "test"]
    test_artists = set(te["artist"])
    cnt = {c: sp[c][(sp[c].role == "addon")].groupby(["artist", "style"]).size() for c in ["emf_AG", "emf_A"]}
    assert cnt["emf_AG"].sort_index().equals(cnt["emf_A"].sort_index()), "per-artist add-on support differs"
    for c in ["emf_AG", "emf_A"]:
        side = sp[c][sp[c].part != "test"]
        assert test_artists <= set(side["artist"]), f"{c}: some test artist unseen"
    for c in ["emf_G", "emf_none"]:
        side = sp[c][sp[c].part != "test"]
        assert not (test_artists & set(side["artist"])), f"{c}: test artist seen"
    for c in ["emf_A", "emf_none"]:
        side = sp[c][sp[c].part != "test"]
        assert not (set(G_te) & set(side["genre"])), f"{c}: test genre seen"
    for c in ["emf_AG", "emf_G"]:
        side = sp[c][sp[c].part != "test"]
        assert set(side["genre"]) & set(te["genre"]), f"{c}: no test genre seen"
    r.update(n_test=len(te), n_test_artists=len(test_artists),
             n_train=int((sp["emf_AG"].part != "test").sum()),
             n_addon=int((sp["emf_AG"].role == "addon").sum()),
             addon_artists_AG=int(cnt["emf_AG"].index.get_level_values(0).nunique()))
    return r
