import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from stylebench.evaluation.classifiers import knn_predict, logreg_fit_predict, ncc_predict
from stylebench.evaluation.metrics import confusion, metrics_from_confusion


def data(seed=0, n=600, d=16, K=4):
    r = np.random.default_rng(seed)
    y = r.integers(K, size=n)
    X = r.normal(size=(n, d)) + 2 * np.eye(K, d)[y]
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32), y


def test_metrics_match_sklearn():
    r = np.random.default_rng(1)
    y, p = r.integers(5, size=300), r.integers(5, size=300)
    m = metrics_from_confusion(confusion(y, p, 5))
    assert np.isclose(m["balanced_accuracy"], balanced_accuracy_score(y, p))
    assert np.isclose(m["macro_f1"], f1_score(y, p, average="macro"))


def test_knn_matches_sklearn():
    X, y = data()
    tr, te = slice(0, 400), slice(400, None)
    p = knn_predict(X[tr], y[tr], X[te], [1, 5], 4)
    for k in [1, 5]:
        m = KNeighborsClassifier(k, metric="cosine").fit(X[tr], y[tr])
        sk = m.predict(X[te])
        lab = y[tr][m.kneighbors(X[te])[1]]
        cnt = np.stack([(lab == c).sum(1) for c in range(4)], 1)
        tie = (cnt == cnt.max(1, keepdims=True)).sum(1) > 1
        # identical whenever the vote is not tied (ties: we use summed similarity, sklearn the lowest label)
        assert (p[k] == sk)[~tie].all()


def test_ncc_and_logreg_learn():
    X, y = data()
    tr, te = slice(0, 400), slice(400, None)
    assert (ncc_predict(X[tr], y[tr], X[te], 4) == y[te]).mean() > 0.7
    (p,) = logreg_fit_predict(X[tr], y[tr], [X[te]], 4, 1e-4)
    assert (p == y[te]).mean() > 0.7
