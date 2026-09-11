"""Probability calibration evaluated without exposing holdout labels to a calibrator."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

CalibrationMethod = Literal["sigmoid", "isotonic"]


def _validate_inputs(y_true: np.ndarray, raw_pd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    raw_pd = np.asarray(raw_pd, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(raw_pd):
        raise ValueError("targets and probabilities must have the same non-zero length")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("targets must contain both target classes")
    if (~np.isfinite(raw_pd)).any() or ((raw_pd < 0) | (raw_pd > 1)).any():
        raise ValueError("probabilities must be finite values between zero and one")
    return y_true, raw_pd


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


@dataclass(frozen=True)
class ProbabilityCalibrator:
    method: CalibrationMethod
    estimator: Any

    def predict(self, raw_pd: np.ndarray) -> np.ndarray:
        raw_pd = np.asarray(raw_pd, dtype=float)
        if (~np.isfinite(raw_pd)).any() or ((raw_pd < 0) | (raw_pd > 1)).any():
            raise ValueError("probabilities must be finite values between zero and one")
        if self.method == "sigmoid":
            result = self.estimator.predict_proba(_logit(raw_pd).reshape(-1, 1))[:, 1]
        else:
            result = self.estimator.predict(raw_pd)
        return np.clip(np.asarray(result, dtype=float), 0, 1)


def fit_calibrator(
    method: CalibrationMethod, y_true: np.ndarray, raw_pd: np.ndarray
) -> ProbabilityCalibrator:
    """Fit one calibration mapping using binary targets and raw probabilities."""
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be sigmoid or isotonic")
    y_true, raw_pd = _validate_inputs(y_true, raw_pd)
    if method == "sigmoid":
        estimator = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
        estimator.fit(_logit(raw_pd).reshape(-1, 1), y_true)
    else:
        estimator = IsotonicRegression(out_of_bounds="clip")
        estimator.fit(raw_pd, y_true)
    return ProbabilityCalibrator(method=method, estimator=estimator)


def cross_fitted_calibration(
    y_true: np.ndarray,
    raw_pd: np.ndarray,
    method: CalibrationMethod,
    folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Produce calibrated probabilities from calibrators that did not see each row's label."""
    y_true, raw_pd = _validate_inputs(y_true, raw_pd)
    if folds < 2:
        raise ValueError("folds must be at least two")
    if min(np.bincount(y_true.astype(int))) < folds:
        raise ValueError("each target class must contain at least one row per fold")

    calibrated = np.full(len(raw_pd), np.nan)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, holdout_idx in splitter.split(raw_pd, y_true):
        calibrator = fit_calibrator(method, y_true[train_idx], raw_pd[train_idx])
        calibrated[holdout_idx] = calibrator.predict(raw_pd[holdout_idx])
    if (~np.isfinite(calibrated)).any():
        raise RuntimeError("cross-fitted calibration did not score every row")
    return calibrated
