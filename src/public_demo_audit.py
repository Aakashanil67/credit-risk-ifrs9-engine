"""Regenerate the offline fairness diagnostic for the served public-demo model."""

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


def _markdown_table(report: pd.DataFrame) -> list[str]:
    lines = [
        "| group | n | observed default rate | mean predicted PD | Brier | ECE (10 bins) | "
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


def write_fairness_audit(
    gender_report: pd.DataFrame, age_report: pd.DataFrame, threshold: float, out_path: Path
) -> None:
    """Write group diagnostics with the correct scope and explicit limits."""
    lines = [
        "# Fairness audit: public-demo PD model",
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
        "Differences in approval, error and calibration rates are signals for investigation, not "
        "proof of cause or fairness. Removing a direct gender feature does not rule out proxy "
        "effects. This is not a disparate-impact assessment, legal review, or production "
        "fairness-monitoring programme. A real assessment would require local applicants, "
        "realised outcomes, policy context, and governance review.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_threshold_analysis(
    metrics: ThresholdMetrics,
    calibration_intercept: float,
    calibration_slope: float,
    out_path: Path,
) -> None:
    """Record the observed test-fold effect of the illustrative threshold separately from policy."""
    lines = [
        "# Decision threshold",
        "",
        "The service uses an illustrative expected-value threshold, not the population default "
        "rate: applications at or above the threshold are declined. With a 12% performing margin, "
        "2% operating cost, 2% capital cost, and 45% LGD, the break-even PD is "
        f"{metrics.threshold:.6f}.",
        "",
        "## Untouched test-fold operating view",
        "",
        f"- Approval rate: **{metrics.approval_rate:.2%}**",
        f"- Default recall among declined applications: **{metrics.recall:.2%}**",
        f"- Observed default precision among declined applications: **{metrics.precision:.2%}**",
        f"- Confusion matrix (actual default positive): TP {metrics.true_positives:,}, FP "
        f"{metrics.false_positives:,}, TN {metrics.true_negatives:,}, FN {metrics.false_negatives:,}.",
        f"- Calibration intercept: **{calibration_intercept:.4f}**; calibration slope: "
        f"**{calibration_slope:.4f}**.",
        "",
        "The numbers describe one historical competition split and are not a lending policy, an "
        "approval recommendation, or evidence of profitability. A lender would estimate product "
        "pricing, capital, LGD, prepayment, collections, and constraints from its own portfolio "
        "before approving any threshold.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_fairness_audit(
        gender_report,
        age_report,
        threshold=policy.threshold,
        out_path=REPORTS_DIR / "fairness_audit.md",
    )
    operating_metrics = threshold_metrics(
        test[TARGET_COL].to_numpy(), predictions, threshold=policy.threshold
    )
    calibration_intercept, calibration_slope = calibration_parameters(
        test[TARGET_COL].to_numpy(), predictions
    )
    write_threshold_analysis(
        operating_metrics,
        calibration_intercept,
        calibration_slope,
        REPORTS_DIR / "threshold_analysis.md",
    )
    print(f"wrote fairness_audit.md for {len(test):,} test rows")


if __name__ == "__main__":
    main()
