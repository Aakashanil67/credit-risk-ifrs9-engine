import pytest

from src.validation_report import build_validation_report


def _sources() -> tuple[dict, dict, dict, dict]:
    return (
        {"model_version": "1.2.0", "test_metrics": {"AUC": 0.67}},
        {
            "metric_intervals": {"auc": {"estimate": 0.67, "lower": 0.66, "upper": 0.68}},
            "gender_approval_gap": {"estimate": 0.08, "lower": 0.07, "upper": 0.09},
            "threshold_sensitivity": [
                {"threshold": 0.10, "approval_rate": 0.70, "recall": 0.50, "precision": 0.25},
                {"threshold": 0.14, "approval_rate": 0.85, "recall": 0.30, "precision": 0.20},
            ],
        },
        {"data_scope": "development_oof_only", "nomination": None, "candidates": []},
        {
            "reference_scope": "frozen_test_reference_window",
            "reference_rows": 800,
            "replay_rows": 200,
            "batches": [{"name": "baseline_replay", "overall_status": "green"}],
        },
    )


def test_validation_report_separates_model_version_from_service_release() -> None:
    report = build_validation_report(*_sources())

    assert "Model version: **1.2.0**" in report
    assert "Service release: **1.3.0**" in report
    assert "incumbent remains preferred" in report
    assert "simulated monitoring" in report.lower()
    assert "Reference rows: **800**" in report
    assert "Replay rows: **200**" in report
    assert "frozen_test_reference_window" in report
    assert "not an independent validation" in report.lower()
    assert "8.00 percentage points" in report
    assert "70.00%" in report


@pytest.mark.parametrize(
    "source_index, key",
    [(0, "model_version"), (1, "metric_intervals"), (2, "data_scope"), (3, "batches")],
)
def test_validation_report_rejects_incomplete_source_data(source_index: int, key: str) -> None:
    sources = list(_sources())
    sources[source_index].pop(key)

    with pytest.raises(ValueError, match=key):
        build_validation_report(*sources)
