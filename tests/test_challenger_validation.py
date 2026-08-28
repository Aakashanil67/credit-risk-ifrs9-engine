import numpy as np
import pandas as pd

from src.challenger_validation import (
    calibration_gate,
    candidate_parent,
    derived_feature_gate,
    lightgbm_oof_predictions,
    render_report,
)
from src.uncertainty import MetricInterval


def interval(estimate: float, lower: float, upper: float) -> MetricInterval:
    return MetricInterval(estimate, lower, upper, 0.95, 100)


def test_calibration_gate_requires_material_brier_improvement_and_no_ranking_change():
    accepted, reasons = calibration_gate(
        {
            "brier": interval(-0.0007, -0.0009, -0.0005),
            "log_loss": interval(-0.001, -0.002, -0.0001),
            "auc": interval(0.0001, -0.0002, 0.0004),
        }
    )

    assert accepted is True
    assert reasons == []


def test_derived_gate_rejects_an_auc_gain_whose_interval_crosses_zero():
    accepted, reasons = derived_feature_gate(
        {
            "auc": interval(0.004, -0.001, 0.008),
            "brier": interval(0.0001, -0.0001, 0.0003),
        }
    )

    assert accepted is False
    assert any("AUC interval" in reason for reason in reasons)


def test_report_calls_the_study_development_only_and_does_not_claim_promotion():
    report = render_report(
        {
            "data_scope": "development_oof_only",
            "incumbent_model_version": "1.2.0",
            "development_rows": 100,
            "candidates": [],
            "nomination": None,
        }
    )

    assert "development-only" in report
    assert "test fold was not used" in report
    assert "incumbent remains preferred" in report


def test_lightgbm_oof_predictions_cover_each_row_deterministically():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"income": rng.normal(size=100), "credit": rng.normal(size=100)})
    y = pd.Series(np.array([0, 1] * 50))

    first = lightgbm_oof_predictions(
        X, y, params={"learning_rate": 0.05, "num_leaves": 7}, n_estimators=5
    )
    second = lightgbm_oof_predictions(
        X, y, params={"learning_rate": 0.05, "num_leaves": 7}, n_estimators=5
    )

    np.testing.assert_allclose(first, second)
    assert first.shape == (100,)
    assert np.isfinite(first).all()
    assert ((0 <= first) & (first <= 1)).all()


def test_candidate_parent_names_the_actual_uncalibrated_parent():
    assert candidate_parent("base_sigmoid") == "base_raw"
    assert candidate_parent("base_isotonic") == "base_raw"
    assert candidate_parent("derived_sigmoid") == "derived_raw"
    assert candidate_parent("derived_isotonic") == "derived_raw"
