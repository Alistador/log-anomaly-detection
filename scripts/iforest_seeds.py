"""Refit Isolation Forest with a few seeds and show how much the budgeted results move.

    python -m scripts.iforest_seeds --features data/features
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from logad.detectors import IForestDetector
from logad.eval import at_threshold, budget_threshold, span_days


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--budgets", type=int, nargs="+", default=[20, 500, 10000])
    args = ap.parse_args(argv)

    train = np.load(args.features / "train.npz")
    test = np.load(args.features / "test.npz")
    train_days, test_days = span_days(train["start"]), span_days(test["start"])

    print("| seed | AP | " + " | ".join(f"{b}/day: alerts, precision, recall" for b in args.budgets) + " |")
    print("|---|---|" + "---|" * len(args.budgets))
    for seed in args.seeds:
        det = IForestDetector(seed=seed).fit(train["X"])
        scores, train_scores = det.score(test["X"]), det.score(train["X"])
        cells = []
        for b in args.budgets:
            r = at_threshold(scores, test["y"], budget_threshold(train_scores, b, train_days),
                             test_days)
            cells.append(f"{r['alerts']}, {r['precision']:.2f}, {r['recall']:.4f}")
        ap_ = average_precision_score(test["y"], scores)
        print(f"| {seed} | {ap_:.3f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
