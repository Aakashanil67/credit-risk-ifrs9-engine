import numpy as np
import pytest

from src.evaluation import binary_metrics, threshold_metrics


def test_binary_metrics_match_a_perfect_ranking_with_nonzero_probability_error():
    """AUC and Brier measure different things: ranking can be perfect but not perfectly calibrated."""
    metrics = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.10, 0.20, 0.80, 0.90]))

    assert metrics.auc == pytest.approx(1.0)
    assert metrics.gini == pytest.approx(1.0)
    assert metrics.ks == pytest.approx(1.0)
    assert metrics.brier == pytest.approx(0.025)


def test_threshold_metrics_count_declines_as_predicted_defaults():
    """At a 50% PD cut-off, the two high-risk applicants decline and both actually default."""
    metrics = threshold_metrics(
        np.array([0, 0, 1, 1]), np.array([0.10, 0.20, 0.80, 0.90]), threshold=0.50
    )

    assert metrics.approval_rate == pytest.approx(0.50)
    assert metrics.true_positives == 2
    assert metrics.false_positives == 0
    assert metrics.true_negatives == 2
    assert metrics.false_negatives == 0
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
