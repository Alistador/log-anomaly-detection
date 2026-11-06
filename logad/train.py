"""Fit both detectors on the training period and keep their training scores, which
eval uses to set alert thresholds without looking at the test period.

    python -m logad.train --features data/features --out models
"""

import argparse
from pathlib import Path

import joblib
import numpy as np

from logad.detectors import DETECTORS


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--out", type=Path, default=Path("models"))
    args = ap.parse_args(argv)

    train = np.load(args.features / "train.npz")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, cls in DETECTORS.items():
        det = cls().fit(train["X"])
        joblib.dump(det, args.out / f"{name}.joblib")
        np.save(args.out / f"{name}_train_scores.npy", det.score(train["X"]))
        print(f"{name}: fitted on {len(train['X'])} blocks")


if __name__ == "__main__":
    main()
