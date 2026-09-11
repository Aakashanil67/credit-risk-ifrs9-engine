import json

import pandas as pd

from src.evaluation import ThresholdMetrics
from src.public_demo_audit import (
    write_fairness_audit,
    write_public_demo_audit_json,
    write_threshold_analysis,
)
from src.uncertainty import MetricInterval


def _interval(estimate: float) -> MetricInterval:
    return MetricInterval(
        estimate=estimate,
        lower=estimate - 0.01,
        upper=estimate + 0.01,
        confidence_level=0.95,
        n_bootstrap=1_000,
    )


def test_fairness_audit_labels_the_public_demo_and_its_limits(tmp_path) -> None:
    gender = pd.DataFrame(
        {
            "n": [10, 12],
            "default_rate": [0.08, 0.10],
            "mean_predicted_pd": [0.09, 0.11],
            "brier": [0.07, 0.08],
            "ece_10": [0.01, 0.02],
            "approval_rate": [0.9, 0.8],
            "tpr": [0.2, 0.3],
            "fpr": [0.1, 0.2],
        },
        index=["F", "M"],
    )
    age = gender.copy()
    age.index = ["18–24", "25–34"]
    out_path = tmp_path / "fairness_audit.md"

    write_fairness_audit(
        gender,
        age,
        threshold=0.14,
        gender_approval_gap=_interval(0.10),
        out_path=out_path,
    )

    text = out_path.read_text(encoding="utf-8")
    assert "public-demo" in text
    assert "not a disparate-impact assessment" in text
    assert "0.140000" in text
    assert "10.00%" in text
    assert "1,000" in text


def test_threshold_report_includes_observed_operating_metrics(tmp_path) -> None:
    metrics = ThresholdMetrics(
        threshold=0.14,
        approval_rate=0.85,
        recall=0.30,
        precision=0.20,
        true_positives=30,
        false_positives=120,
        true_negatives=800,
        false_negatives=70,
    )
    out_path = tmp_path / "threshold_analysis.md"

    write_threshold_analysis(
        metrics,
        0.08,
        1.02,
        {"auc": _interval(0.67), "brier": _interval(0.07)},
        out_path,
    )

    text = out_path.read_text(encoding="utf-8")
    assert "85.00%" in text
    assert "30" in text
    assert "not a lending policy" in text
    assert "95% stratified bootstrap interval" in text
    assert "0.6600" in text


def test_public_demo_audit_json_contains_aggregate_intervals_only(tmp_path) -> None:
    out_path = tmp_path / "public_demo_audit.json"

    write_public_demo_audit_json(
        model_version="1.2.0",
        test_rows=100,
        metric_intervals={"auc": _interval(0.67)},
        threshold_metrics=ThresholdMetrics(0.14, 0.85, 0.30, 0.20, 30, 120, 800, 70),
        calibration_intercept=0.08,
        calibration_slope=1.02,
        gender_approval_gap=_interval(0.10),
        out_path=out_path,
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["model_version"] == "1.2.0"
    assert report["metric_intervals"]["auc"]["n_bootstrap"] == 1_000
    assert "SK_ID_CURR" not in json.dumps(report)
