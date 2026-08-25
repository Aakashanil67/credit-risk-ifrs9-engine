import numpy as np
import pytest

from src.fairness import group_fairness_metrics


def test_group_fairness_reports_error_and_approval_rates_by_audit_group():
    """Protected values stay in offline diagnostics, where reviewers can inspect disparate outcomes."""
    report = group_fairness_metrics(
        y_true=np.array([0, 1, 0, 1]),
        pd_score=np.array([0.05, 0.30, 0.10, 0.40]),
        groups=np.array(["F", "F", "M", "M"]),
        threshold=0.20,
    )

    female = report.loc["F"]
    male = report.loc["M"]
    assert female["n"] == 2
    assert female["approval_rate"] == pytest.approx(0.5)
    assert male["default_rate"] == pytest.approx(0.5)
    assert {"brier", "ece_10", "tpr", "fpr", "precision"}.issubset(report.columns)
