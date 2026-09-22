"""Offline group diagnostics for the payment-difficulty risk model."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def expected_calibration_error(y_true: np.ndarray, pd_score: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.digitize(pd_score, edges[1:-1], right=True)
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - pd_score[mask].mean())
    return float(error)


def calibration_parameters(y_true: np.ndarray, pd_score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    logit = np.log(np.clip(pd_score, 1e-6, 1 - 1e-6) / np.clip(1 - pd_score, 1e-6, 1))
    model = LogisticRegression(C=1e6, solver="lbfgs").fit(logit.reshape(-1, 1), y_true)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def group_fairness_metrics(
    y_true: np.ndarray, pd_score: np.ndarray, groups: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Measure calibration and decision rates per protected or audit-only group."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")

    records = []
    for group in pd.unique(groups):
        mask = groups == group
        outcomes = y_true[mask]
        scores = pd_score[mask]
        predicted_default = scores >= threshold
        actual_default = outcomes == 1
        true_positive = int((predicted_default & actual_default).sum())
        false_positive = int((predicted_default & ~actual_default).sum())
        false_negative = int((~predicted_default & actual_default).sum())
        calibration_intercept, calibration_slope = calibration_parameters(outcomes, scores)
        records.append(
            {
                "group": group,
                "n": int(mask.sum()),
                "default_rate": float(outcomes.mean()),
                "mean_predicted_pd": float(scores.mean()),
                "brier": float(np.mean((outcomes - scores) ** 2)),
                "ece_10": expected_calibration_error(outcomes, scores),
                "calibration_intercept": calibration_intercept,
                "calibration_slope": calibration_slope,
                "approval_rate": float((~predicted_default).mean()),
                "tpr": float(true_positive / (true_positive + false_negative))
                if true_positive + false_negative
                else np.nan,
                "fpr": float(false_positive / (~actual_default).sum())
                if (~actual_default).sum()
                else np.nan,
                "precision": float(true_positive / (true_positive + false_positive))
                if true_positive + false_positive
                else np.nan,
            }
        )
    return pd.DataFrame.from_records(records).set_index("group").sort_index()
