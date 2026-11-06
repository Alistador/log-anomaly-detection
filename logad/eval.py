"""Evaluate both detectors on the test period under an alert budget.

A budget of B alerts/day becomes a score threshold using the training period only:
the threshold is the score that the top B * (training days) training blocks exceed.
That threshold is then applied to the test period, so the realised alert rate can
drift away from B, which is what would happen in production too.

    python -m logad.eval --features data/features --models models --out reports
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from logad.detectors import DETECTORS  # noqa: E402

DAY = 86400.0
BUDGETS = [20, 100, 500, 2000, 10000]


def span_days(start: np.ndarray) -> float:
    return (start.max() - start.min()) / DAY


def budget_threshold(train_scores: np.ndarray, per_day: float, train_days: float) -> float:
    """Score that at most per_day * train_days training blocks exceed. Ties at the
    threshold are not alerted on, so the budget is never overshot on training data."""
    k = int(round(per_day * train_days))
    s = np.sort(train_scores)[::-1]
    if k <= 0:
        return float(s[0])
    if k >= len(s):
        return float(-np.inf)
    return float(s[k])


def at_threshold(scores: np.ndarray, y: np.ndarray, threshold: float, days: float) -> dict:
    alert = scores > threshold
    n = int(alert.sum())
    tp = int((alert & (y == 1)).sum())
    return {
        "alerts": n,
        "alerts_per_day": n / days,
        "true_alerts": tp,
        "precision": tp / n if n else float("nan"),
        "recall": tp / int(y.sum()),
    }


def evaluate(models: Path, train, test, budgets=BUDGETS) -> dict:
    train_days, test_days = span_days(train["start"]), span_days(test["start"])
    results = {"train_days": train_days, "test_days": test_days,
               "test_blocks": int(len(test["y"])), "test_anomalies": int(test["y"].sum()),
               "detectors": {}}
    for name in DETECTORS:
        det = joblib.load(models / f"{name}.joblib")
        train_scores = np.load(models / f"{name}_train_scores.npy")
        scores = det.score(test["X"])
        rows = []
        for b in budgets:
            thr = budget_threshold(train_scores, b, train_days)
            rows.append({"budget_per_day": b, "threshold": thr,
                         **at_threshold(scores, test["y"], thr, test_days)})
        results["detectors"][name] = {
            "average_precision": float(average_precision_score(test["y"], scores)),
            "budgets": rows,
            "scores": scores,
        }
    return results


def plot_pr(results: dict, y: np.ndarray, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, r in results["detectors"].items():
        p, rec, _ = precision_recall_curve(y, r["scores"])
        line, = ax1.plot(rec, p, label=f"{name} (AP {r['average_precision']:.3f})")
        b = [row for row in r["budgets"] if row["alerts"]]
        ax1.scatter([row["recall"] for row in b], [row["precision"] for row in b],
                    color=line.get_color(), s=18, zorder=3)
        x = [row["alerts_per_day"] for row in b]
        ax2.plot(x, [row["precision"] for row in b], "o-", color=line.get_color(),
                 label=f"{name} precision")
        ax2.plot(x, [row["recall"] for row in b], "s--", color=line.get_color(),
                 label=f"{name} recall")
    base = y.mean()
    ax1.axhline(base, color="grey", lw=0.8, ls=":")
    ax1.text(0.99, base, f"base rate {base:.1%}", ha="right", va="bottom", fontsize=8,
             color="grey")
    ax1.set(xlabel="recall", ylabel="precision", title="Test period, dots = budget thresholds",
            xlim=(0, 1), ylim=(0, 1.02))
    ax1.legend(loc="lower left")
    ax2.set_xscale("log")
    ax2.set(xlabel="alerts per day in test period (log)", ylim=(0, 1.02),
            title="Loosening the budget")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_scores_over_time(results: dict, test, name: str, out: Path) -> None:
    r = results["detectors"][name]
    hours = (test["start"] - test["start"].min()) / 3600
    scores = r["scores"] - r["scores"].min() + 1e-6
    y = test["y"] == 1
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(hours[~y], scores[~y], s=1, color="#9aa4ad", label="normal block",
               rasterized=True)
    ax.scatter(hours[y], scores[y], s=2, color="#d1495b", label="labelled anomaly",
               rasterized=True)
    offset = r["scores"].min() - 1e-6
    for row, ls in zip(r["budgets"][:3], ["-", "--", ":"]):
        ax.axhline(row["threshold"] - offset, color="black", lw=0.8, ls=ls,
                   label=f"{row['budget_per_day']}/day threshold")
    ax.set_yscale("log")
    ax.set(xlabel="hours into test period", ylabel=f"{name} score (shifted, log)",
           title=f"{name} score per block over the test period")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def markdown_tables(results: dict) -> str:
    lines = [
        f"Test period: {results['test_days'] * 24:.1f} h, {results['test_blocks']} blocks, "
        f"{results['test_anomalies']} labelled anomalies.",
        "",
        "| detector | budget/day | alerts | alerts/day (test) | true alerts | precision | recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results["detectors"].items():
        for row in r["budgets"]:
            lines.append(
                f"| {name} | {row['budget_per_day']} | {row['alerts']} | "
                f"{row['alerts_per_day']:.0f} | {row['true_alerts']} | "
                f"{row['precision']:.3f} | {row['recall']:.4f} |"
            )
    lines += ["", "| detector | average precision (test) |", "|---|---|"]
    for name, r in results["detectors"].items():
        lines.append(f"| {name} | {r['average_precision']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--plot-detector", default="pca", choices=list(DETECTORS))
    args = ap.parse_args(argv)

    train = np.load(args.features / "train.npz")
    test = np.load(args.features / "test.npz")
    results = evaluate(args.models, train, test)

    args.out.mkdir(parents=True, exist_ok=True)
    plot_pr(results, test["y"], args.out / "pr_curve.png")
    plot_scores_over_time(results, test, args.plot_detector, args.out / "score_over_time.png")
    tables = markdown_tables(results)
    (args.out / "results.md").write_text(tables, encoding="utf-8")
    for r in results["detectors"].values():
        del r["scores"]
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(tables)


if __name__ == "__main__":
    main()
