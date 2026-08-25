"""Evaluation measures for frozen PD models."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    auc: float
    gini: float
    ks: float
    brier: float


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    approval_rate: float
    recall: float
    precision: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


def binary_metrics(y_true: np.ndarray, pd_score: np.ndarray) -> BinaryMetrics:
    """Measure discrimination and probability error without choosing a cut-off."""
    auc = float(roc_auc_score(y_true, pd_score))
    return BinaryMetrics(
        auc=auc,
        gini=2 * auc - 1,
        ks=float(ks_2samp(pd_score[y_true == 1], pd_score[y_true == 0]).statistic),
        brier=float(brier_score_loss(y_true, pd_score)),
    )


def threshold_metrics(
    y_true: np.ndarray, pd_score: np.ndarray, threshold: float
) -> ThresholdMetrics:
    """Report the operational effect of declining applicants at or above a PD threshold."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")

    predicted_default = pd_score >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted_default, labels=[0, 1]).ravel()
    predicted_default_count = int(predicted_default.sum())

    return ThresholdMetrics(
        threshold=threshold,
        approval_rate=float((~predicted_default).mean()),
        recall=float(tp / (tp + fn)) if tp + fn else 0.0,
        precision=float(tp / predicted_default_count) if predicted_default_count else 0.0,
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
    )
