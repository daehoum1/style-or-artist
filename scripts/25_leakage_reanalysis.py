"""Revision round 2 (C3): leakage-gap association re-analysed (i) with logistic-regression gaps (same classifier
family as the probes) and (ii) with alternative covariates for the partial correlation (Joint accuracy, style
probe), since IID accuracy is itself partly a consequence of artist leakage (possible over-control).
Output: artifacts/results/rev2/leakage_reanalysis.csv"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.config import ARTIFACTS  # noqa: E402
from stylebench.evaluation.statistics import partial_spearman_with_uncertainty, spearman_with_uncertainty  # noqa: E402

rows = []
for d in ["wikiart", "multitask_painting"]:
    E = pd.read_csv(ARTIFACTS / "final" / d / "encoder_level.csv", index_col=0)
    M = pd.read_csv(ARTIFACTS / "final" / d / "table2_main_all.csv")
    lr = M[(M.evaluator == "logreg") & (M.metric == "balanced_accuracy")].set_index("encoder")
    E["lr_iid"], E["lr_artist_gap"] = lr.loc[E.index, "iid_mean"], lr.loc[E.index, "artist_gap"]
    E["lr_genre_gap"], E["lr_joint"] = lr.loc[E.index, "genre_gap"], lr.loc[E.index, "joint_mean"]
    E["lr_joint_gap"] = lr.loc[E.index, "joint_gap"]
    tests = [("within_style_artist", "artist_gap", "knn"), ("within_style_artist", "lr_artist_gap", "logreg"),
             ("within_style_genre", "genre_gap", "knn"), ("within_style_genre", "lr_genre_gap", "logreg"),
             ("combined_within_style", "joint_gap", "knn"), ("combined_within_style", "lr_joint_gap", "logreg")]
    for leak, gap, clf in tests:
        r = spearman_with_uncertainty(E[leak], E[gap])
        row = {"dataset": d, "leak": leak, "gap": gap, "classifier": clf, "rho": r["rho"], "ci_lo": r["ci_lo"],
               "ci_hi": r["ci_hi"], "perm_p": r["perm_p"]}
        for cov in (["iid", "joint", "style"] if clf == "knn" else ["lr_iid", "lr_joint", "style"]):
            pr = partial_spearman_with_uncertainty(E[leak], E[gap], E[cov])
            key = cov.replace("lr_", "")
            row.update({f"partial_{key}": pr["rho"], f"partial_{key}_lo": pr["ci_lo"], f"partial_{key}_hi": pr["ci_hi"],
                        f"partial_{key}_p": pr["perm_p"]})
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(ARTIFACTS / "results" / "rev2" / "leakage_reanalysis.csv", index=False)
print(df.round(2).to_string())
