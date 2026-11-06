import numpy as np

from logad.eval import at_threshold, budget_threshold


def test_budget_threshold_never_overshoots_with_ties():
    scores = np.array([5, 4, 4, 4, 1, 0], dtype=float)
    # budget of 2 alerts: the tie at 4 means only the 5 can be alerted on
    thr = budget_threshold(scores, per_day=2, train_days=1)
    assert (scores > thr).sum() == 1
    thr = budget_threshold(scores, per_day=4, train_days=1)
    assert (scores > thr).sum() == 4


def test_budget_scales_with_days():
    scores = np.arange(100, dtype=float)
    thr = budget_threshold(scores, per_day=10, train_days=2.0)
    assert (scores > thr).sum() == 20


def test_at_threshold():
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    y = np.array([1, 0, 1, 1])
    r = at_threshold(scores, y, threshold=0.5, days=0.5)
    assert r["alerts"] == 3 and r["alerts_per_day"] == 6
    assert r["true_alerts"] == 2
    assert r["precision"] == 2 / 3 and r["recall"] == 2 / 3
