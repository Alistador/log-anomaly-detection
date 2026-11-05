"""Turn per-block traces into event count vectors and split them by time.

Blocks are ordered by the timestamp of their first line. The earlier part of the time
range is the training period, the rest is the test period. Labels are attached here
but the detectors never see them during training.

    python -m logad.features --parsed data/parsed --labels data/raw/anomaly_label.csv --out data/features
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def event_vocab(templates_csv: Path) -> list[str]:
    ids = pd.read_csv(templates_csv)["event_id"].tolist()
    return sorted(ids, key=lambda e: int(e[1:]))


def count_matrix(sequences, vocab: list[str]) -> np.ndarray:
    """sequences: iterable of space-separated event id strings, one per block."""
    index = {e: i for i, e in enumerate(vocab)}
    sequences = list(sequences)
    X = np.zeros((len(sequences), len(vocab)), dtype=np.int32)
    for row, seq in enumerate(sequences):
        for e in seq.split():
            X[row, index[e]] += 1
    return X


def load_blocks(parsed: Path, labels_csv: Path) -> pd.DataFrame:
    traces = pd.read_csv(parsed / "traces.csv")
    labels = pd.read_csv(labels_csv).rename(columns={"BlockId": "block_id", "Label": "label"})
    df = traces.merge(labels, on="block_id", how="inner")
    missing = len(traces) - len(df)
    if missing:
        print(f"warning: {missing} blocks have no label and are dropped")
    df["y"] = (df["label"] == "Anomaly").astype(np.int8)
    return df.sort_values("start", kind="stable").reset_index(drop=True)


def time_split(start: np.ndarray, train_frac: float) -> float:
    """Cutoff timestamp so that the first train_frac of the time range is training."""
    return float(start.min() + train_frac * (start.max() - start.min()))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parsed", type=Path, default=Path("data/parsed"))
    ap.add_argument("--labels", type=Path, default=Path("data/raw/anomaly_label.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/features"))
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="fraction of the time range used for training")
    args = ap.parse_args(argv)

    vocab = event_vocab(args.parsed / "templates.csv")
    df = load_blocks(args.parsed, args.labels)
    X = count_matrix(df["events"], vocab)
    cutoff = time_split(df["start"].to_numpy(), args.train_frac)
    is_train = (df["start"] < cutoff).to_numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, mask in [("train", is_train), ("test", ~is_train)]:
        part = df[mask]
        np.savez_compressed(
            args.out / f"{name}.npz",
            X=X[mask], y=part["y"].to_numpy(), start=part["start"].to_numpy(),
            block_id=part["block_id"].to_numpy().astype(str),
        )
        hours = (part["start"].max() - part["start"].min()) / 3600
        print(f"{name}: {mask.sum()} blocks, {part['y'].sum()} anomalies "
              f"({part['y'].mean():.2%}), {hours:.1f} h")
    (args.out / "vocab.json").write_text(json.dumps(vocab))


if __name__ == "__main__":
    main()
