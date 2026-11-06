"""The two detectors. Both are unsupervised: fit on unlabeled count vectors, higher score
means more anomalous.

PCA follows Xu et al. (SOSP 2009): TF-IDF weight the event counts, project onto the
principal subspace that explains most of the variance, and score each block by the
squared distance to that subspace (squared prediction error, SPE).

Isolation Forest scores log-scaled counts by how quickly random splits isolate a block.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class PCADetector:
    name = "pca"

    def __init__(self, variance: float = 0.95):
        self.variance = variance

    def _weight(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float64) * self.idf_

    def fit(self, X: np.ndarray) -> "PCADetector":
        n = X.shape[0]
        df = np.count_nonzero(X, axis=0)
        # events never seen in training get weight 0 instead of dividing by zero
        self.idf_ = np.where(df > 0, np.log(n / np.maximum(df, 1)), 0.0)
        W = self._weight(X)
        self.mean_ = W.mean(axis=0)
        self.pca_ = PCA(n_components=self.variance, svd_solver="full").fit(W - self.mean_)
        self.components_ = self.pca_.components_
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Y = self._weight(X) - self.mean_
        residual = Y - (Y @ self.components_.T) @ self.components_
        return np.einsum("ij,ij->i", residual, residual)


class IForestDetector:
    name = "iforest"

    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X: np.ndarray) -> "IForestDetector":
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1
        ).fit(np.log1p(X))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model_.score_samples(np.log1p(X))


DETECTORS = {d.name: d for d in (PCADetector, IForestDetector)}
