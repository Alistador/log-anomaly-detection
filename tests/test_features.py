import numpy as np
import pandas as pd

from logad.features import count_matrix, load_blocks, time_split


def test_count_matrix():
    X = count_matrix(["E1 E2 E1", "", "E3"], ["E1", "E2", "E3"])
    assert X.tolist() == [[2, 1, 0], [0, 0, 0], [0, 0, 1]]


def test_time_split():
    assert time_split(np.array([100, 200, 300]), 0.5) == 200


def test_load_blocks_sorts_and_labels(tmp_path):
    pd.DataFrame({"block_id": ["b", "a", "c"], "start": [5, 1, 9], "end": [6, 2, 9],
                  "events": ["E1", "E2", "E1 E1"]}).to_csv(tmp_path / "traces.csv", index=False)
    pd.DataFrame({"BlockId": ["a", "b"], "Label": ["Normal", "Anomaly"]}).to_csv(
        tmp_path / "labels.csv", index=False)
    df = load_blocks(tmp_path, tmp_path / "labels.csv")
    assert df["block_id"].tolist() == ["a", "b"]  # unlabelled block dropped
    assert df["y"].tolist() == [0, 1]
