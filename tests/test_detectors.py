import numpy as np
import pytest

from logad.detectors import DETECTORS


def synthetic(seed=0, n=3000, n_anomalies=15):
    """Normal blocks mix a few optional events; anomalies (0.5%, about HDFS's rate
    divided by six) contain an event normal blocks never have and lack a usual one."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n + n_anomalies, 6), dtype=np.int32)
    X[:, 0] = 3
    X[:, 1] = rng.integers(0, 3, len(X))
    X[:, 2] = 3
    X[:, 3] = rng.binomial(1, 0.4, len(X)) * 2
    X[:, 4] = rng.binomial(1, 0.6, len(X))
    X[n:, 2] = 0
    X[n:, 5] = 1
    y = np.r_[np.zeros(n), np.ones(n_anomalies)]
    return X, y


@pytest.mark.parametrize("name", list(DETECTORS))
def test_anomalies_score_higher(name):
    X, y = synthetic()
    scores = DETECTORS[name]().fit(X).score(X)
    top = np.argsort(scores)[::-1][: int(y.sum())]
    assert y[top].mean() > 0.9


def test_pca_ignores_events_unseen_in_training():
    X, y = synthetic()
    det = DETECTORS["pca"]().fit(X[y == 0])  # column 5 is all zero here
    assert det.idf_[5] == 0
    assert np.isfinite(det.score(X)).all()
