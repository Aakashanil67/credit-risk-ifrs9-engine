import pandas as pd
import pytest

from src.config import TARGET_COL
from src.monitoring_demo import apply_stress, render_monitoring_report, split_monitoring_windows


def sample_batch() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100.0] * 20,
            "AMT_CREDIT": [200.0] * 20,
            "OCCUPATION_TYPE": ["Laborers"] * 20,
        }
    )


def test_mild_stress_is_deterministic_and_does_not_mutate_source():
    source = sample_batch()
    first = apply_stress(source, "mild_shift", seed=42)
    second = apply_stress(source, "mild_shift", seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert source["AMT_INCOME_TOTAL"].eq(100).all()
    assert first["AMT_INCOME_TOTAL"].eq(90).all()
    assert first["AMT_CREDIT"].eq(210).all()
    assert first["OCCUPATION_TYPE"].isna().sum() == 2


def test_severe_stress_uses_the_declared_25_percent_missingness():
    stressed = apply_stress(sample_batch(), "severe_shift", seed=42)
    assert stressed["AMT_INCOME_TOTAL"].eq(70).all()
    assert stressed["AMT_CREDIT"].eq(240).all()
    assert stressed["OCCUPATION_TYPE"].isna().sum() == 5


def test_monitoring_report_labels_replay_as_simulated_not_production():
    markdown = render_monitoring_report(
        {
            "reference_scope": "frozen_test_reference_window",
            "reference_rows": 800,
            "replay_rows": 200,
            "model_version": "1.2.0",
            "batches": [
                {
                    "name": "baseline_replay",
                    "transformations": {},
                    "monitoring": {
                        "overall_status": "green",
                        "score": {"psi": 0.0},
                        "approval": {"rate": 0.9},
                        "performance": {"auc": 0.7, "brier": 0.07},
                        "features": {},
                    },
                }
            ],
        }
    )
    assert "simulated monitoring" in markdown.lower()
    assert "not production observations" in markdown.lower()
    assert "out-of-sample monitoring reference" in markdown.lower()
    assert "development reference" not in markdown.lower()


def test_monitoring_windows_are_deterministic_stratified_and_disjoint():
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": range(1_000),
            TARGET_COL: [0] * 800 + [1] * 200,
            "AMT_INCOME_TOTAL": [100.0] * 1_000,
            "AMT_CREDIT": [200.0] * 1_000,
            "OCCUPATION_TYPE": ["Sales staff"] * 1_000,
        }
    )
    first_reference, first_replay = split_monitoring_windows(frame, replay_rows=200, seed=42)
    second_reference, second_replay = split_monitoring_windows(frame, replay_rows=200, seed=42)

    assert len(first_reference) == 800
    assert len(first_replay) == 200
    assert set(first_reference["SK_ID_CURR"]).isdisjoint(first_replay["SK_ID_CURR"])
    assert first_reference[TARGET_COL].mean() == pytest.approx(0.20)
    assert first_replay[TARGET_COL].mean() == pytest.approx(0.20)
    pd.testing.assert_frame_equal(first_reference, second_reference)
    pd.testing.assert_frame_equal(first_replay, second_replay)


@pytest.mark.parametrize(
    "replay_rows",
    [0, 1_000],
)
def test_monitoring_windows_reject_invalid_replay_size(replay_rows):
    frame = pd.DataFrame({TARGET_COL: [0, 1]})

    with pytest.raises(ValueError, match="replay_rows"):
        split_monitoring_windows(frame, replay_rows=replay_rows)


def test_monitoring_windows_reject_missing_target():
    with pytest.raises(ValueError, match="TARGET"):
        split_monitoring_windows(pd.DataFrame({"income": [1.0, 2.0]}), replay_rows=1)


def test_monitoring_windows_reject_single_class_target():
    frame = pd.DataFrame({TARGET_COL: [0, 0, 0], "income": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="both target classes"):
        split_monitoring_windows(frame, replay_rows=1)
