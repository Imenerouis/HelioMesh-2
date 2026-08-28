"""
HelioMesh -- Real Telemetry Anomaly Detector
=============================================
Trains an unsupervised anomaly detector on the NORMAL portion of the
benchmark dataset (labels == 0) and scores all timesteps.

Two detectors are available; the active one is chosen by the
``METHOD`` constant:

  "zscore"      -- rolling z-score on each channel (no external deps)
  "iforest"     -- Isolation Forest (scikit-learn)

The detector is kept in memory only (no frozen pkl) because the
training data changes depending on whether the real SMAP download
succeeded.

IMPORTANT: this model predicts NORMAL vs ANOMALOUS over raw telemetry
channels.  It has NO relationship to HelioMesh simulation states
(NOMINAL / STANDBY / SAFE_MODE / CRITICAL_AHEAD).
"""

import numpy as np

METHOD = "iforest"  # "zscore" | "iforest"

# Rolling window for z-score method
_WINDOW = 20

# Isolation Forest contamination prior (fraction expected anomalous)
_CONTAMINATION = 0.10

# Decision threshold on normalised anomaly score (0-1)
# Applied uniformly so callers can use raw scores and pick their own cutoff.
THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Z-score detector
# ---------------------------------------------------------------------------
class _ZScoreDetector:
    """Per-channel rolling z-score, averaged across channels."""

    def __init__(self, window: int = _WINDOW):
        self.window = window
        self._train_mean: np.ndarray | None = None
        self._train_std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        """
        X: shape (n_normal, n_channels) -- normal-only training data.
        Computes global per-channel statistics over the training set.
        """
        self._train_mean = X.mean(axis=0)
        self._train_std  = X.std(axis=0)
        self._train_std[self._train_std == 0] = 1e-6  # avoid div-by-zero

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores in [0, 1].  Higher = more anomalous.
        Uses a rolling window z-score smoothed over ``self.window`` steps,
        then normalised by the max observed z-score (clamped to >=1).
        """
        z = np.abs((X - self._train_mean) / self._train_std)
        mean_z = z.mean(axis=1)  # shape (T,)

        # Rolling mean smoothing
        smoothed = np.convolve(mean_z, np.ones(self.window) / self.window, mode="same")

        # Normalise to [0, 1]
        z_max = max(smoothed.max(), 1.0)
        scores = np.clip(smoothed / z_max, 0.0, 1.0)
        return scores.astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
        scores = self.score_samples(X)
        return (scores >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Isolation Forest detector
# ---------------------------------------------------------------------------
class _IForestDetector:
    """Isolation Forest wrapper."""

    def __init__(self, contamination: float = _CONTAMINATION, seed: int = 42):
        from sklearn.ensemble import IsolationForest

        self._model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> None:
        """X: shape (n_normal, n_channels)"""
        self._model.fit(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores in [0, 1].  Higher = more anomalous.
        IsolationForest.score_samples() returns negative average path lengths
        (lower = more anomalous).  We negate and normalise to [0, 1].
        """
        raw = self._model.score_samples(X)  # shape (T,) -- negative values
        # negate so higher means more anomalous
        inverted = -raw
        lo, hi = inverted.min(), inverted.max()
        if hi == lo:
            return np.zeros(len(X), dtype=np.float32)
        scores = (inverted - lo) / (hi - lo)
        return scores.astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
        scores = self.score_samples(X)
        return (scores >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def build_detector(method: str = METHOD) -> "_ZScoreDetector | _IForestDetector":
    """Return an un-fitted detector instance."""
    if method == "zscore":
        return _ZScoreDetector()
    elif method == "iforest":
        return _IForestDetector()
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'zscore' or 'iforest'.")


def train_detector(dataset: dict, method: str = METHOD):
    """
    Fit a detector on the normal-only portion of *dataset*.

    Parameters
    ----------
    dataset : dict
        As returned by dataset.load_dataset().
    method : str
        "zscore" or "iforest".

    Returns
    -------
    detector  : fitted detector instance
    scores    : np.ndarray shape (T,) -- anomaly scores for full dataset
    y_pred    : np.ndarray shape (T,) -- binary predictions (0/1)
    """
    X      = dataset["data"]       # shape (T, C)
    labels = dataset["labels"]     # shape (T,)

    X_train = X[labels == 0]
    print(
        f"  [anomaly_model] method={method}  "
        f"train_size={len(X_train)} (normal only)  "
        f"test_size={len(X)} (full)"
    )

    detector = build_detector(method)
    detector.fit(X_train)

    scores = detector.score_samples(X)
    y_pred = detector.predict(X)

    return detector, scores, y_pred
