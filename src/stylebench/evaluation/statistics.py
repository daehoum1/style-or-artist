"""Statistics helpers: bootstrap CIs, paired tests, rank correlations with bootstrap/permutation, partial Spearman."""
from __future__ import annotations

import numpy as np
from scipy import stats


def boot_ci(x, n_boot=10000, alpha=0.05, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    b = stat(x[r.integers(len(x), size=(n_boot, len(x)))], axis=1)
    return tuple(np.quantile(b, [alpha / 2, 1 - alpha / 2]))


def paired_summary(a, b, seed=0) -> dict:
    """a - b over paired seeds: mean, sd, 95% bootstrap CI, Wilcoxon signed-rank p, #seeds a>b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = boot_ci(d, seed=seed)
    p = stats.wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
    return {"mean": d.mean(), "sd": d.std(ddof=1) if len(d) > 1 else np.nan, "ci_lo": lo, "ci_hi": hi,
            "wilcoxon_p": p, "n_pos": int((d > 0).sum()), "n": len(d)}


def spearman_with_uncertainty(x, y, n_boot=10000, n_perm=10000, seed=0) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = stats.spearmanr(x, y).statistic
    r = np.random.default_rng(seed)
    n = len(x)
    bs = []
    for _ in range(n_boot):
        i = r.integers(n, size=n)
        if len(np.unique(x[i])) > 2 and len(np.unique(y[i])) > 2:
            bs.append(stats.spearmanr(x[i], y[i]).statistic)
    perm = np.array([stats.spearmanr(x, r.permutation(y)).statistic for _ in range(n_perm)])
    return {"rho": rho, "ci_lo": np.nanquantile(bs, 0.025), "ci_hi": np.nanquantile(bs, 0.975),
            "perm_p": (np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (n_perm + 1), "n": n}


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y controlling for z (rank-transform, then partial Pearson)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        A = np.c_[np.ones_like(b), b]
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def partial_spearman_with_uncertainty(x, y, z, n_boot=10000, n_perm=10000, seed=0) -> dict:
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    rho = partial_spearman(x, y, z)
    r = np.random.default_rng(seed)
    n = len(x)
    bs = []
    for _ in range(n_boot):
        i = r.integers(n, size=n)
        if min(len(np.unique(v[i])) for v in (x, y, z)) > 3:
            bs.append(partial_spearman(x[i], y[i], z[i]))
    perm = np.array([partial_spearman(x, r.permutation(y), z) for _ in range(n_perm)])
    return {"rho": rho, "ci_lo": np.nanquantile(bs, 0.025), "ci_hi": np.nanquantile(bs, 0.975),
            "perm_p": (np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (n_perm + 1), "n": n}


def kendall_tau(a, b) -> float:
    return float(stats.kendalltau(a, b).statistic)
