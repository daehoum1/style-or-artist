"""Style evaluation for (dataset, encoders) on all saved split protocols."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylebench.features.cache import available  # noqa: E402
from stylebench.evaluation.style_eval import run_style_eval  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("encoders", nargs="+")
ap.add_argument("--split_root", default=None)
ap.add_argument("--evaluators", default="knn5,knn_sel,ncc,logreg")
ap.add_argument("--protocols", default="iid,artist_disjoint,genre_disjoint,joint,mf_iid,mf_artist,mf_genre,mf_joint")
a = ap.parse_args()
for e in a.encoders:
    if not available(a.dataset, e):
        print(f"[missing embeddings] {a.dataset}/{e}")
        continue
    sroot = Path(__file__).resolve().parents[1] / "artifacts" / "splits" / (a.split_root or a.dataset)
    protos = [p for p in a.protocols.split(",") if (sroot / p).exists()]
    df = run_style_eval(a.dataset, e, protos, a.split_root, evaluators=a.evaluators.split(","))
    s = df.groupby(["protocol", "evaluator"])[["accuracy", "balanced_accuracy", "macro_f1"]].mean().round(4)
    print(f"== {a.dataset} / {e}\n{s.unstack('evaluator')['balanced_accuracy'].to_string()}", flush=True)
