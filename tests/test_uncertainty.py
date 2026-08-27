import numpy as np
import pytest

from src.uncertainty import (
    bootstrap_binary_metric_intervals,
    paired_bootstrap_metric_deltas,
)


def test_bootstrap_intervals_are_deterministic_and_contain_point_estimates():
    y = np.array([0] * 80 + [1] * 20)
    pd_score = np.linspace(0.01, 0.99, 100)

    first = bootstrap_binary_metric_intervals(y, pd_score, n_bootstrap=100, seed=42)
    second = bootstrap_binary_metric_intervals(y, pd_score, n_bootstrap=100, seed=42)

    assert first == second
    assert set(first) == {"auc", "gini", "ks", "brier", "pr_auc", "log_loss"}
    assert first["auc"].lower <= first["auc"].estimate <= first["auc"].upper
    assert first["brier"].confidence_level == pytest.approx(0.95)


def test_paired_delta_preserves_direction_for_a_better_challenger():
    y = np.array([0] * 80 + [1] * 20)
    incumbent = np.where(y == 1, 0.35, 0.15)
    challenger = np.where(y == 1, 0.80, 0.05)

    delta = paired_bootstrap_metric_deltas(y, incumbent, challenger, n_bootstrap=100, seed=42)

    assert delta["auc"].estimate >= 0
    assert delta["brier"].estimate < 0
    assert delta["log_loss"].estimate < 0


def test_bootstrap_rejects_misaligned_or_single_class_inputs():
    with pytest.raises(ValueError, match="same non-zero length"):
        bootstrap_binary_metric_intervals(np.array([0, 1]), np.array([0.2]))
    with pytest.raises(ValueError, match="both target classes"):
        bootstrap_binary_metric_intervals(np.array([0, 0]), np.array([0.1, 0.2]))
