import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring import (
    build_monitoring_reference,
    evaluate_monitoring_batch,
    load_monitoring_reference,
    population_stability_index,
    save_monitoring_reference,
    status_for_signal,
    status_for_unseen,
)


def test_psi_is_zero_for_identical_distributions_and_positive_for_shift():
    reference = np.array([0.50, 0.50])
    assert population_stability_index(reference, reference) == pytest.approx(0.0)
    assert population_stability_index(reference, np.array([0.80, 0.20])) > 0.1


def test_monitoring_reference_is_aggregate_json_without_rows(tmp_path):
    X = pd.DataFrame(
        {
            "income": [10.0, 20.0, 30.0, 40.0],
            "segment": pd.Series(["A", "A", "B", None], dtype="category"),
        }
    )
    reference = build_monitoring_reference(
        X, pd_score=np.array([0.1, 0.2, 0.3, 0.4]), threshold=0.25, model_version="1.2.0"
    )
    path = tmp_path / "reference.json"

    save_monitoring_reference(reference, path)
    restored = load_monitoring_reference(path)

    assert restored == reference
    assert restored["row_count"] == 4
    assert set(restored["features"]) == {"income", "segment"}
    assert "rows" not in json.dumps(restored).lower()


def test_status_boundaries_include_red_precedence_and_strict_unseen_amber():
    assert status_for_signal(0.0999, amber=0.10, red=0.25) == "green"
    assert status_for_signal(0.10, amber=0.10, red=0.25) == "amber"
    assert status_for_signal(0.25, amber=0.10, red=0.25) == "red"
    assert status_for_unseen(0.0, red=0.01) == "green"
    assert status_for_unseen(0.0001, red=0.01) == "amber"
    assert status_for_unseen(0.01, red=0.01) == "red"


def test_batch_monitor_reports_drift_quality_and_outcome_metrics():
    reference_X = pd.DataFrame(
        {"income": np.arange(1, 101, dtype=float), "segment": pd.Categorical(["A", "B"] * 50)}
    )
    reference_pd = np.linspace(0.01, 0.50, 100)
    y = np.array([0] * 80 + [1] * 20)
    reference = build_monitoring_reference(reference_X, reference_pd, 0.20, "1.2.0", y_true=y)
    current_X = reference_X.astype({"segment": "object"}).copy()
    current_X.loc[:19, "segment"] = None
    current_pd = np.clip(reference_pd + 0.20, 0, 1)

    result = evaluate_monitoring_batch(reference, current_X, current_pd, y_true=y)

    assert result["row_count"] == 100
    assert result["overall_status"] in {"amber", "red"}
    assert result["features"]["segment"]["missing_rate"] == pytest.approx(0.20)
    assert {"auc", "brier", "gini", "ks", "pr_auc", "log_loss"} <= set(result["performance"])


def test_batch_monitor_rejects_missing_columns():
    reference = build_monitoring_reference(
        pd.DataFrame({"income": [1.0, 2.0], "segment": ["A", "B"]}),
        pd_score=np.array([0.1, 0.2]),
        threshold=0.15,
        model_version="1.2.0",
    )
    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_monitoring_batch(
            reference, pd.DataFrame({"income": [1.0, 2.0]}), np.array([0.1, 0.2])
        )
