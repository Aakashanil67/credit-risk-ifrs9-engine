"""Regenerate offline diagnostics for the served public-demo model."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import load_artifact_bundle
from src.config import (
    CAPITAL_COST_RATE,
    DEFAULT_LGD,
    OPERATING_COST_RATE,
    PERFORMING_MARGIN_RATE,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COL,
    model_bundle_dir,
)
from src.data_loader import load_application_data
from src.decision_policy import DecisionPolicy
from src.evaluation import ThresholdMetrics, threshold_metrics
from src.fairness import calibration_parameters, group_fairness_metrics
from src.features import build_lgbm_features
from src.model_profiles import ModelProfile
from src.preprocessing import split_data
from src.uncertainty import (
    MetricInterval,
    bootstrap_binary_metric_intervals,
    bootstrap_group_rate_difference,
)


def _markdown_table(report: pd.DataFrame) -> list[str]:
    lines = [
        "| group | n | observed event rate | mean predicted risk | Brier | ECE (10 bins) | "
        "approval rate | TPR | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, row in report.iterrows():
        lines.append(
            f"| {group} | {row['n']:,.0f} | {row['default_rate']:.2%} | "
            f"{row['mean_predicted_pd']:.2%} | {row['brier']:.4f} | {row['ece_10']:.4f} | "
            f"{row['approval_rate']:.2%} | {row['tpr']:.2%} | {row['fpr']:.2%} |"
        )
    return lines


def build_threshold_sensitivity(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: tuple[float, ...],
) -> list[ThresholdMetrics]:
    """Calculate descriptive operating metrics at pre-specified thresholds."""
    return [threshold_metrics(y_true, scores, threshold=value) for value in thresholds]


def write_fairness_audit(
    gender_report: pd.DataFrame,
    age_report: pd.DataFrame,
    threshold: float,
    gender_approval_gap: MetricInterval,
    out_path: Path,
) -> None:
    """Write group diagnostics with the correct scope and explicit limits."""
    lines = [
        "# Fairness audit: public-demo risk model",
        "",
        "This audit evaluates the 15-field public-demo LightGBM model on its untouched Home "
        "Credit test fold. `CODE_GENDER` is retained only for this offline diagnostic; it is not "
        "a model feature and the API does not accept it. The illustrative decision threshold is "
        f"{threshold:.6f}: applications below it are approved.",
        "",
        "## Gender groups",
        "",
        *_markdown_table(gender_report),
        "",
        "## Age bands",
        "",
        *_markdown_table(age_report),
        "",
        "## Gender approval-rate difference",
        "",
        "The estimate below is the female approval rate minus the male approval rate. Its 95% "
        "stratified bootstrap interval uses "
        f"{gender_approval_gap.n_bootstrap:,} deterministic resamples of this historical test fold.",
        "",
        f"- Estimate: **{100 * gender_approval_gap.estimate:.2f} percentage points**",
        f"- 95% stratified bootstrap interval: **{100 * gender_approval_gap.lower:.2f} to "
        f"{100 * gender_approval_gap.upper:.2f} percentage points**",
        "",
        "In this test fold, the gender groups differ in observed event mix, average model score, "
        "approval rate, and error rates. Those differences are descriptive. They do not identify "
        "whether the model, correlated input variables, the historical sample, or another factor "
        "caused the gap.",
        "",
        "Differences in approval, error and calibration rates are signals for investigation, not "
        "proof of cause or fairness. Removing a direct gender feature does not rule out proxy "
        "effects. This is not a disparate-impact assessment, legal review, or production "
        "fairness-monitoring programme. A real assessment would require local applicants, "
        "realised outcomes, policy context, and governance review.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_threshold_analysis(
    metrics: ThresholdMetrics,
    sensitivity_metrics: list[ThresholdMetrics],
    calibration_intercept: float,
    calibration_slope: float,
    metric_intervals: dict[str, MetricInterval],
    out_path: Path,
) -> None:
    """Record the observed test-fold effect of the illustrative threshold separately from policy."""
    lines = [
        "# Decision threshold",
        "",
        "The service uses an illustrative expected-value threshold, not the population default "
        "rate: applications at or above the threshold are declined. With a 12% performing margin, "
        "2% operating cost, 2% capital cost, and 45% LGD, the illustrative break-even risk "
        "threshold is "
        f"{metrics.threshold:.6f}.",
        "",
        "## Untouched test-fold operating view",
        "",
        f"- Approval rate: **{metrics.approval_rate:.2%}**",
        f"- Payment-difficulty event capture among declined applications: **{metrics.recall:.2%}**",
        f"- Observed event rate among declined applications: **{metrics.precision:.2%}**",
        f"- Confusion matrix (actual default positive): TP {metrics.true_positives:,}, FP "
        f"{metrics.false_positives:,}, TN {metrics.true_negatives:,}, FN {metrics.false_negatives:,}.",
        f"- Calibration intercept: **{calibration_intercept:.4f}**; calibration slope: "
        f"**{calibration_slope:.4f}**.",
        "",
        "## Fixed-threshold sensitivity",
        "",
        "These fixed operating points are descriptive. They were not searched to select or "
        "change the deployed threshold.",
        "",
        "| threshold | approval rate | event capture among declined | observed event rate among declined |",
        "|---:|---:|---:|---:|",
        *[
            f"| {row.threshold:.6f} | {row.approval_rate:.2%} | {row.recall:.2%} | "
            f"{row.precision:.2%} |"
            for row in sensitivity_metrics
        ],
        "",
        "## Test-fold model uncertainty",
        "",
        "Intervals are 95% stratified bootstrap intervals from the untouched historical test "
        "fold; they quantify sampling uncertainty, not future portfolio performance.",
        "",
        "| metric | estimate | 95% stratified bootstrap interval |",
        "|---|---:|---:|",
        *[
            f"| {name.replace('_', ' ')} | {interval.estimate:.4f} | "
            f"{interval.lower:.4f} to {interval.upper:.4f} |"
            for name, interval in metric_intervals.items()
        ],
        "",
        "The numbers describe one historical competition split and are not a lending policy, an "
        "approval recommendation, or evidence of profitability. A lender would estimate product "
        "pricing, capital, LGD, prepayment, collections, and constraints from its own portfolio "
        "before approving any threshold.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_public_demo_audit_json(
    model_version: str,
    test_rows: int,
    metric_intervals: dict[str, MetricInterval],
    threshold_metrics: ThresholdMetrics,
    threshold_sensitivity: list[ThresholdMetrics],
    calibration_intercept: float,
    calibration_slope: float,
    gender_approval_gap: MetricInterval,
    out_path: Path,
) -> None:
    """Write only aggregate audit diagnostics for downstream validation reporting."""
    report = {
        "model_version": model_version,
        "test_rows": test_rows,
        "seed": RANDOM_SEED,
        "bootstrap_samples": gender_approval_gap.n_bootstrap,
        "metric_intervals": {name: asdict(interval) for name, interval in metric_intervals.items()},
        "threshold_metrics": asdict(threshold_metrics),
        "threshold_sensitivity": [asdict(row) for row in threshold_sensitivity],
        "calibration": {
            "intercept": calibration_intercept,
            "slope": calibration_slope,
        },
        "gender_approval_gap": asdict(gender_approval_gap),
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _align_categories(X: pd.DataFrame, category_dtypes: dict) -> pd.DataFrame:
    aligned = X.copy()
    for column, dtype in category_dtypes.items():
        aligned[column] = aligned[column].astype(dtype)
    return aligned


def main() -> None:
    profile = ModelProfile.PUBLIC_DEMO
    bundle = load_artifact_bundle(model_bundle_dir(profile.value))
    df = load_application_data()
    _train, _val, test = split_data(df, seed=RANDOM_SEED)
    test_X = _align_categories(build_lgbm_features(test, profile=profile), bundle.category_dtypes)
    predictions = bundle.model.predict_proba(test_X)[:, 1]
    policy = DecisionPolicy(
        margin_rate=PERFORMING_MARGIN_RATE,
        operating_cost_rate=OPERATING_COST_RATE,
        capital_cost_rate=CAPITAL_COST_RATE,
        lgd=DEFAULT_LGD,
    )

    gender_report = group_fairness_metrics(
        test[TARGET_COL].to_numpy(), predictions, test["CODE_GENDER"].to_numpy(), policy.threshold
    )
    ages = -test["DAYS_BIRTH"] / 365.25
    age_groups = pd.cut(
        ages,
        bins=[18, 25, 35, 45, 55, 65, np.inf],
        right=False,
        labels=["18–24", "25–34", "35–44", "45–54", "55–64", "65+"],
    ).astype(str)
    age_report = group_fairness_metrics(
        test[TARGET_COL].to_numpy(), predictions, age_groups.to_numpy(), policy.threshold
    )
    metric_intervals = bootstrap_binary_metric_intervals(
        test[TARGET_COL].to_numpy(), predictions, n_bootstrap=1_000, seed=RANDOM_SEED
    )
    approved = predictions < policy.threshold
    gender_approval_gap = bootstrap_group_rate_difference(
        approved,
        test["CODE_GENDER"].to_numpy(),
        "F",
        "M",
        n_bootstrap=1_000,
        seed=RANDOM_SEED,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_fairness_audit(
        gender_report,
        age_report,
        threshold=policy.threshold,
        gender_approval_gap=gender_approval_gap,
        out_path=REPORTS_DIR / "fairness_audit.md",
    )
    operating_metrics = threshold_metrics(
        test[TARGET_COL].to_numpy(), predictions, threshold=policy.threshold
    )
    sensitivity_metrics = build_threshold_sensitivity(
        test[TARGET_COL].to_numpy(),
        predictions,
        thresholds=(0.10, policy.threshold, 0.18),
    )
    calibration_intercept, calibration_slope = calibration_parameters(
        test[TARGET_COL].to_numpy(), predictions
    )
    write_threshold_analysis(
        operating_metrics,
        sensitivity_metrics,
        calibration_intercept,
        calibration_slope,
        metric_intervals,
        REPORTS_DIR / "threshold_analysis.md",
    )
    write_public_demo_audit_json(
        model_version=bundle.metadata["model_version"],
        test_rows=len(test),
        metric_intervals=metric_intervals,
        threshold_metrics=operating_metrics,
        threshold_sensitivity=sensitivity_metrics,
        calibration_intercept=calibration_intercept,
        calibration_slope=calibration_slope,
        gender_approval_gap=gender_approval_gap,
        out_path=REPORTS_DIR / "public_demo_audit.json",
    )
    print(f"wrote fairness_audit.md for {len(test):,} test rows")


if __name__ == "__main__":
    main()
