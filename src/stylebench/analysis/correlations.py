"""Leakage-vs-gap correlation analysis across encoders."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.statistics import partial_spearman_with_uncertainty, spearman_with_uncertainty


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def correlation_table(enc_df: pd.DataFrame, tests: list[dict], control: str | None = None) -> pd.DataFrame:
    rows = []
    for t in tests:
        d = enc_df[[t["leak"], t["gap"]] + ([control] if control else [])].dropna()
        if len(d) < 4:
            continue
        r = spearman_with_uncertainty(d[t["leak"]], d[t["gap"]])
        row = {"leak": t["leak"], "gap": t["gap"], **r}
        if control:
            pr = partial_spearman_with_uncertainty(d[t["leak"]], d[t["gap"]], d[control])
            row.update({f"partial_rho|{control}": pr["rho"], "partial_ci_lo": pr["ci_lo"],
                        "partial_ci_hi": pr["ci_hi"], "partial_perm_p": pr["perm_p"]})
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["perm_p_holm"] = holm(out["perm_p"])
    return out
