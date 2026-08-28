"""Development-only challenger evaluation for the frozen public-demo PD model."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.artifacts import load_artifact_bundle
from src.calibration import cross_fitted_calibration
from src.challenger_features import add_collectable_ratios
from src.config import RANDOM_SEED, REPORTS_DIR, TARGET_COL, model_bundle_dir
from src.data_loader import load_application_data
from src.evaluation import binary_metrics
from src.features import build_lgbm_features
from src.model_profiles import ModelProfile
from src.preprocessing import split_data
from src.uncertainty import MetricInterval, paired_bootstrap_metric_deltas

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SAMPLE_ROWS = 20_000
CV_FOLDS = 5


def _align_fold_categories(
    train_X: pd.DataFrame, holdout_X: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_X.copy()
    holdout = holdout_X.copy()
    for column in train.select_dtypes("category").columns:
        dtype = train[column].cat.remove_unused_categories().dtype
        train[column] = train[column].astype(dtype)
        holdout[column] = holdout[column].astype(dtype)
    return train, holdout


def lightgbm_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_estimators: int,
    folds: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Score each development row with a fold model that was not fitted on that row."""
    if len(X) == 0 or len(X) != len(y):
        raise ValueError("features and targets must have the same non-zero length")
    if folds < 2 or min(y.value_counts()) < folds:
        raise ValueError("each target class must contain at least one row per fold")

    predictions = np.full(len(X), np.nan)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, holdout_idx in splitter.split(X, y):
        fold_train, fold_holdout = _align_fold_categories(X.iloc[train_idx], X.iloc[holdout_idx])
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            verbose=-1,
            **params,
        )
        model.fit(fold_train, y.iloc[train_idx])
        predictions[holdout_idx] = model.predict_proba(fold_holdout)[:, 1]
    if (~np.isfinite(predictions)).any() or ((predictions < 0) | (predictions > 1)).any():
        raise RuntimeError("out-of-fold model did not produce bounded probabilities for every row")
    return predictions


def calibration_gate(metric_delta: dict[str, MetricInterval]) -> tuple[bool, list[str]]:
    """Apply the predeclared calibration acceptance criteria."""
    reasons = []
    brier = metric_delta["brier"]
    if brier.estimate > -0.0005:
        reasons.append("Brier improvement is smaller than 0.0005")
    if brier.upper >= 0:
        reasons.append("Brier interval does not remain below zero")
    if metric_delta["log_loss"].estimate > 0:
        reasons.append("Log loss worsens")
    if abs(metric_delta["auc"].estimate) > 0.001:
        reasons.append("AUC changes by more than 0.001")
    return not reasons, reasons


def derived_feature_gate(metric_delta: dict[str, MetricInterval]) -> tuple[bool, list[str]]:
    """Apply the predeclared collectable-ratio challenger acceptance criteria."""
    reasons = []
    auc = metric_delta["auc"]
    brier = metric_delta["brier"]
    if auc.estimate < 0.003:
        reasons.append("AUC improvement is smaller than 0.003")
    if auc.lower <= 0:
        reasons.append("AUC interval crosses zero")
    if brier.estimate > 0.0002:
        reasons.append("Brier score worsens by more than 0.0002")
    return not reasons, reasons


def _metric_dict(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    metrics = binary_metrics(y_true, probabilities)
    return {
        "auc": metrics.auc,
        "gini": metrics.gini,
        "ks": metrics.ks,
        "brier": metrics.brier,
        "pr_auc": metrics.pr_auc,
        "log_loss": metrics.log_loss,
    }


def _interval_dict(intervals: dict[str, MetricInterval]) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "estimate": interval.estimate,
            "lower": interval.lower,
            "upper": interval.upper,
            "confidence_level": interval.confidence_level,
            "n_bootstrap": interval.n_bootstrap,
        }
        for name, interval in intervals.items()
    }


def candidate_parent(name: str) -> str:
    """Return the uncalibrated candidate from which a calibration variant was derived."""
    if name.startswith("base_"):
        return "base_raw"
    if name.startswith("derived_"):
        return "derived_raw"
    raise ValueError(f"unsupported candidate name: {name}")


def _candidate(
    name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    parent_probabilities: np.ndarray,
    gate: str,
) -> dict:
    deltas = paired_bootstrap_metric_deltas(
        y_true,
        parent_probabilities,
        probabilities,
        n_bootstrap=BOOTSTRAP_SAMPLES,
        seed=RANDOM_SEED,
        metric_names=("auc", "brier") if gate == "derived" else ("auc", "brier", "log_loss"),
        sample_size=min(BOOTSTRAP_SAMPLE_ROWS, len(y_true)),
    )
    accepted, reasons = (
        calibration_gate(deltas) if gate == "calibration" else derived_feature_gate(deltas)
    )
    return {
        "name": name,
        "metrics": _metric_dict(y_true, probabilities),
        "parent": "base_raw" if gate == "derived" else candidate_parent(name),
        "gate": gate,
        "accepted": accepted,
        "reasons": reasons,
        "metric_delta": _interval_dict(deltas),
    }


def _choose_nominee(candidates: list[dict]) -> str | None:
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    if not accepted:
        return None
    winner = min(
        accepted,
        key=lambda candidate: (
            candidate["metrics"]["brier"],
            -candidate["metrics"]["auc"],
            candidate["metrics"]["log_loss"],
        ),
    )
    return str(winner["name"])


def run_challenger_study() -> dict:
    """Evaluate all six candidates on development-only out-of-fold predictions."""
    profile = ModelProfile.PUBLIC_DEMO
    metadata = load_artifact_bundle(model_bundle_dir(profile.value)).metadata
    train, validation = split_data(load_application_data(), seed=RANDOM_SEED)[:2]
    development = pd.concat([train, validation], ignore_index=True)
    y_true = development[TARGET_COL].to_numpy()
    base_X = build_lgbm_features(development, profile=profile)
    derived_X = add_collectable_ratios(base_X)
    selection = metadata["selection"]

    base_raw = lightgbm_oof_predictions(
        base_X,
        development[TARGET_COL],
        selection["parameters"],
        selection["early_stopping_iteration"],
    )
    derived_raw = lightgbm_oof_predictions(
        derived_X,
        development[TARGET_COL],
        selection["parameters"],
        selection["early_stopping_iteration"],
    )
    base_sigmoid = cross_fitted_calibration(y_true, base_raw, method="sigmoid", folds=CV_FOLDS)
    base_isotonic = cross_fitted_calibration(y_true, base_raw, method="isotonic", folds=CV_FOLDS)
    derived_sigmoid = cross_fitted_calibration(
        y_true, derived_raw, method="sigmoid", folds=CV_FOLDS
    )
    derived_isotonic = cross_fitted_calibration(
        y_true, derived_raw, method="isotonic", folds=CV_FOLDS
    )

    candidates = [
        _candidate("base_sigmoid", y_true, base_sigmoid, base_raw, "calibration"),
        _candidate("base_isotonic", y_true, base_isotonic, base_raw, "calibration"),
        _candidate("derived_raw", y_true, derived_raw, base_raw, "derived"),
    ]
    derived_gate_result = candidates[-1]
    for name, probabilities in (
        ("derived_sigmoid", derived_sigmoid),
        ("derived_isotonic", derived_isotonic),
    ):
        calibrated = _candidate(name, y_true, probabilities, derived_raw, "calibration")
        calibrated["accepted"] = bool(calibrated["accepted"] and derived_gate_result["accepted"])
        if not derived_gate_result["accepted"]:
            calibrated["reasons"] = [
                *calibrated["reasons"],
                "Derived-feature parent did not pass its acceptance gate",
            ]
        candidates.append(calibrated)

    return {
        "data_scope": "development_oof_only",
        "incumbent_model_version": metadata["model_version"],
        "development_rows": len(development),
        "seed": RANDOM_SEED,
        "folds": CV_FOLDS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_sample_rows": min(BOOTSTRAP_SAMPLE_ROWS, len(development)),
        "incumbent": {"name": "base_raw", "metrics": _metric_dict(y_true, base_raw)},
        "candidates": candidates,
        "nomination": _choose_nominee(candidates),
    }


def render_report(result: dict) -> str:
    """Render a reviewer-readable report without serialising row-level predictions."""
    lines = [
        "# Development-only challenger validation",
        "",
        "This development-only study compares candidates using five-fold out-of-fold predictions on train plus "
        "validation data. The test fold was not used for candidate selection or this challenger "
        "report. It is evidence for a later review, not a model promotion.",
        "",
        f"- Incumbent model version: **{result['incumbent_model_version']}**",
        f"- Development rows: **{result['development_rows']:,}**",
        f"- Folds / seed / bootstrap samples: **{result.get('folds', 5)} / "
        f"{result.get('seed', RANDOM_SEED)} / {result.get('bootstrap_samples', BOOTSTRAP_SAMPLES):,}**",
        f"- Paired-bootstrap sample rows: **{result.get('bootstrap_sample_rows', 'full sample')}**",
        "",
        "| candidate | AUC | Gini | Brier | Log loss | gate | result |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    incumbent = result.get("incumbent")
    if incumbent:
        metrics = incumbent["metrics"]
        lines.append(
            f"| base_raw | {metrics['auc']:.4f} | {metrics['gini']:.4f} | "
            f"{metrics['brier']:.4f} | {metrics['log_loss']:.4f} | incumbent | retained |"
        )
    for candidate in result["candidates"]:
        metrics = candidate["metrics"]
        lines.append(
            f"| {candidate['name']} | {metrics['auc']:.4f} | {metrics['gini']:.4f} | "
            f"{metrics['brier']:.4f} | {metrics['log_loss']:.4f} | {candidate['gate']} | "
            f"{'pass' if candidate['accepted'] else 'fail'} |"
        )
    lines += ["", "## Predeclared gate results", ""]
    for candidate in result["candidates"]:
        message = (
            "Passed every applicable gate."
            if candidate["accepted"]
            else "; ".join(candidate["reasons"])
        )
        lines.append(f"- **{candidate['name']}:** {message}")
    lines += ["", "## Outcome", ""]
    if result["nomination"] is None:
        lines.append("No candidate passed every predeclared gate; the incumbent remains preferred.")
    else:
        lines.append(
            f"`{result['nomination']}` is nominated for a separate promotion review; it is not deployed by this study."
        )
    lines += [
        "",
        "The study uses stratified m-out-of-n paired bootstrap samples for candidate gate intervals; "
        "the full development sample is still used for each point estimate. The study is not nested "
        "cross-validation: the incumbent's frozen parameters and tree count "
        "come from the prior v1.2 development process. The OOF protocol prevents an estimator or "
        "calibrator from scoring a row it fitted, but this report is still development evidence rather "
        "than an independent validation.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(result: dict, reports_dir: Path = REPORTS_DIR) -> None:
    """Persist aggregate challenger evidence in JSON and Markdown formats."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "challenger_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (reports_dir / "challenger_validation.md").write_text(render_report(result), encoding="utf-8")


def main() -> None:
    result = run_challenger_study()
    write_outputs(result)
    print(f"wrote challenger report for {result['development_rows']:,} development rows")


if __name__ == "__main__":
    main()
