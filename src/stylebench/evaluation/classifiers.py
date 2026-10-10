"""GPU implementations of the lightweight evaluators on frozen, L2-normalized embeddings."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _t(x, dtype=torch.float32, device=None):
    return torch.as_tensor(x, dtype=dtype, device=device or DEV)


def _lr_device(n, d, k):
    """Small problems are kernel-launch bound on the GPU; solve them on the CPU (same model and minimizer)."""
    return "cpu" if n * k * d < 5e7 else DEV


def knn_predict(Xtr, ytr, Xte, ks, K, chunk=4096):
    """Cosine k-NN majority vote for several k at once; ties broken by larger summed similarity.
    Returns dict k -> predictions."""
    A, B = _t(Xtr), _t(Xte)
    y = _t(ytr, torch.long)
    kmax = min(max(ks), len(ytr))
    out = {k: [] for k in ks}
    for i in range(0, len(B), chunk):
        S = B[i:i + chunk] @ A.T
        sim, idx = S.topk(kmax, dim=1)
        lab = y[idx]
        for k in ks:
            kk = min(k, kmax)
            votes = torch.zeros(len(lab), K, device=DEV)
            votes.scatter_add_(1, lab[:, :kk], torch.ones_like(sim[:, :kk]))
            ssum = torch.zeros(len(lab), K, device=DEV)
            ssum.scatter_add_(1, lab[:, :kk], sim[:, :kk])
            out[k].append((votes + 1e-3 * ssum).argmax(1).cpu())  # sim in [-1,1], 1e-3*k < 1 vote
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def ncc_predict(Xtr, ytr, Xte, K):
    A, B, y = _t(Xtr), _t(Xte), _t(ytr, torch.long)
    C = torch.zeros(K, A.shape[1], device=DEV).index_add_(0, y, A)
    present = torch.bincount(y, minlength=K) > 0
    C = F.normalize(C, dim=1)
    S = B @ C.T
    S[:, ~present] = -1e9
    return S.argmax(1).cpu().numpy()


def logreg_fit_predict(Xtr, ytr, Xte_list, K, wd, max_iter=300, seed=0, init=None, return_params=False):
    """Multinomial logistic regression, full-batch L-BFGS, L2 penalty wd*||W||^2 (bias unpenalized).
    Inputs standardized with training statistics."""
    torch.manual_seed(seed)
    dev = _lr_device(len(Xtr), Xtr.shape[1], K)
    A = _t(Xtr, device=dev)
    mu, sd = A.mean(0, keepdim=True), A.std(0, keepdim=True) + 1e-6
    A = (A - mu) / sd
    y = _t(ytr, torch.long, device=dev)
    if init is None:
        W = torch.zeros(A.shape[1], K, device=dev, requires_grad=True)
        b = torch.zeros(K, device=dev, requires_grad=True)
    else:  # warm start (the objective is strictly convex, so the minimizer does not depend on the start)
        W = init[0].detach().to(dev).clone().requires_grad_(True)
        b = init[1].detach().to(dev).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)
    absent = torch.bincount(y, minlength=K) == 0

    def closure():
        opt.zero_grad()
        logits = A @ W + b
        logits = logits.masked_fill(absent, -1e4)
        loss = F.cross_entropy(logits, y) + wd * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    preds = []
    with torch.no_grad():
        for X in Xte_list:
            Z = ((_t(X, device=dev) - mu) / sd) @ W + b
            preds.append(Z.masked_fill(absent, -1e4).argmax(1).cpu().numpy())
    return (preds, (W, b)) if return_params else preds



def logreg_select_and_fit(Xfit, yfit, Xval, yval, Xfull, yfull, Xte, K, wd_grid, max_iter, seed, score_fn):
    """Regularization path from the largest to the smallest wd (each fit warm-started from the previous
    solution), wd chosen by score_fn(y_val, pred) (ties -> larger wd), then refit on the full training side
    warm-started from the selected solution. Returns (test predictions, selected wd, validation score)."""
    best = None
    init = None
    for wd in sorted(wd_grid, reverse=True):
        (pv,), init = logreg_fit_predict(Xfit, yfit, [Xval], K, wd, max_iter, seed, init=init, return_params=True)
        sc = score_fn(yval, pv)
        if best is None or sc > best[1]:
            best = (wd, sc, init)
    wd, sc, params = best
    (pt,) = logreg_fit_predict(Xfull, yfull, [Xte], K, wd, max_iter, seed, init=params)
    return pt, wd, sc
