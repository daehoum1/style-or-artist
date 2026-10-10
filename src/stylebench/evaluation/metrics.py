import numpy as np


def confusion(y, p, K):
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)


def metrics_from_confusion(C: np.ndarray) -> dict:
    """Metrics over classes PRESENT in the evaluation set (rows with support > 0)."""
    sup = C.sum(1)
    pres = sup > 0
    tp = np.diag(C).astype(float)
    rec = np.where(sup > 0, tp / np.maximum(sup, 1), 0.0)
    pc = C.sum(0)
    prec = np.where(pc > 0, tp / np.maximum(pc, 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    K = int(pres.sum())
    bacc = rec[pres].mean()
    return {"accuracy": tp.sum() / C.sum(), "balanced_accuracy": bacc, "macro_f1": f1[pres].mean(),
            "n_classes_eval": K, "n_eval": int(C.sum()),
            "balanced_accuracy_norm": (bacc - 1 / K) / (1 - 1 / K) if K > 1 else np.nan,
            "per_class": {"recall": rec, "precision": prec, "f1": f1, "support": sup}}
