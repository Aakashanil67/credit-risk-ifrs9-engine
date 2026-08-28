"""Aggregate drift and data-quality monitoring for the public-demo model."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MONITORING_EPSILON, MONITORING_PSI_BINS
from src.evaluation import binary_metrics


@dataclass(frozen=True)
class MonitoringThresholds:
    psi_amber: float = 0.10
    psi_red: float = 0.25
    missing_delta_amber: float = 0.05
    missing_delta_red: float = 0.10
    unseen_red: float = 0.01
    approval_delta_amber: float = 0.05
    approval_delta_red: float = 0.10
    auc_drop_amber: float = 0.02
    auc_drop_red: float = 0.05
    brier_increase_amber: float = 0.005
    brier_increase_red: float = 0.010


def _validate_probabilities(pd_score: np.ndarray, expected_length: int) -> np.ndarray:
    scores = np.asarray(pd_score, dtype=float)
    if len(scores) != expected_length or expected_length == 0:
        raise ValueError("scores must match the non-empty batch length")
    if (~np.isfinite(scores)).any() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite probabilities between zero and one")
    return scores


def _normalise(proportions: np.ndarray) -> np.ndarray:
    values = np.asarray(proportions, dtype=float)
    if values.ndim != 1 or len(values) == 0 or (~np.isfinite(values)).any() or (values < 0).any():
        raise ValueError("proportions must be a non-empty finite non-negative vector")
    adjusted = values + MONITORING_EPSILON
    return adjusted / adjusted.sum()


def population_stability_index(reference: np.ndarray, current: np.ndarray) -> float:
    """Calculate PSI with epsilon smoothing for aligned distributions."""
    if len(reference) != len(current):
        raise ValueError("reference and current distributions must have the same length")
    reference = _normalise(reference)
    current = _normalise(current)
    return float(np.sum((current - reference) * np.log(current / reference)))


def status_for_signal(value: float, amber: float, red: float) -> str:
    if red < amber or value < 0:
        raise ValueError("status thresholds must satisfy zero <= amber <= red")
    if value >= red:
        return "red"
    if value >= amber:
        return "amber"
    return "green"


def status_for_unseen(value: float, red: float) -> str:
    if value < 0 or not 0 < red <= 1:
        raise ValueError("unseen category thresholds must satisfy zero < red <= one")
    if value >= red:
        return "red"
    if value > 0:
        return "amber"
    return "green"


def _numeric_edges(series: pd.Series) -> list[float | None]:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return [None, None]
    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, MONITORING_PSI_BINS + 1)))
    inner = quantiles[1:-1].astype(float).tolist()
    return [None, *inner, None]


def _histogram_proportions(values: pd.Series | np.ndarray, edges: list[float | None]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    resolved = np.array(
        [
            -np.inf if edge is None and index == 0 else np.inf if edge is None else edge
            for index, edge in enumerate(edges)
        ],
        dtype=float,
    )
    counts, _ = np.histogram(finite, bins=resolved)
    return _normalise(counts)


def _category_labels(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), "<MISSING>").astype(str)


def _categorical_proportions(labels: pd.Series, categories: list[str]) -> np.ndarray:
    mapped = labels.where(labels.isin(categories), "<UNSEEN>")
    counts = mapped.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    return _normalise(counts)


def _metric_dict(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    metrics = binary_metrics(y_true, scores)
    return {
        "auc": metrics.auc,
        "gini": metrics.gini,
        "ks": metrics.ks,
        "brier": metrics.brier,
        "pr_auc": metrics.pr_auc,
        "log_loss": metrics.log_loss,
    }


def build_monitoring_reference(
    X: pd.DataFrame,
    pd_score: np.ndarray,
    threshold: float,
    model_version: str,
    y_true: np.ndarray | None = None,
) -> dict:
    """Build a serialisable aggregate reference without retaining application rows."""
    if X.empty or X.columns.duplicated().any():
        raise ValueError("reference features must be non-empty with unique columns")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between zero and one")
    scores = _validate_probabilities(pd_score, len(X))
    features = {}
    for column in X.columns:
        series = X[column]
        missing_rate = float(series.isna().mean())
        if pd.api.types.is_numeric_dtype(series):
            edges = _numeric_edges(series)
            features[column] = {
                "kind": "numeric",
                "missing_rate": missing_rate,
                "bin_edges": edges,
                "proportions": _histogram_proportions(series, edges).tolist(),
            }
        else:
            labels = _category_labels(series)
            categories = sorted(set(labels).union({"<MISSING>", "<UNSEEN>"}))
            features[column] = {
                "kind": "categorical",
                "missing_rate": missing_rate,
                "categories": categories,
                "proportions": _categorical_proportions(labels, categories).tolist(),
            }
    performance = None
    if y_true is not None:
        y_true = np.asarray(y_true)
        if len(y_true) != len(X):
            raise ValueError("targets must match the reference feature length")
        performance = _metric_dict(y_true, scores)
    score_edges = _numeric_edges(pd.Series(scores))
    return {
        "schema_version": 1,
        "model_version": model_version,
        "row_count": len(X),
        "decision_threshold": threshold,
        "approval_rate": float((scores < threshold).mean()),
        "performance": performance,
        "score": {
            "bin_edges": score_edges,
            "proportions": _histogram_proportions(scores, score_edges).tolist(),
        },
        "features": features,
    }


def save_monitoring_reference(reference: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(reference, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_monitoring_reference(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _maximum_status(statuses: list[str]) -> str:
    order = {"green": 0, "amber": 1, "red": 2}
    return max(statuses, key=order.__getitem__)


def evaluate_monitoring_batch(
    reference: dict,
    X: pd.DataFrame,
    pd_score: np.ndarray,
    y_true: np.ndarray | None = None,
    thresholds: MonitoringThresholds | None = None,
) -> dict:
    """Compare one batch with an aggregate reference and return aggregate diagnostics only."""
    thresholds = MonitoringThresholds() if thresholds is None else thresholds
    if X.empty or X.columns.duplicated().any():
        raise ValueError("batch features must be non-empty with unique columns")
    feature_reference = reference["features"]
    missing = sorted(set(feature_reference).difference(X.columns))
    if missing:
        raise ValueError(f"batch is missing required columns: {missing}")
    scores = _validate_probabilities(pd_score, len(X))
    feature_results = {}
    statuses = []
    for column, feature in feature_reference.items():
        series = X[column]
        missing_rate = float(series.isna().mean())
        missing_delta = abs(missing_rate - float(feature["missing_rate"]))
        missing_status = status_for_signal(
            missing_delta, thresholds.missing_delta_amber, thresholds.missing_delta_red
        )
        if feature["kind"] == "numeric":
            proportions = _histogram_proportions(series, feature["bin_edges"])
            unseen_rate = 0.0
            unseen_status = "green"
        else:
            labels = _category_labels(series)
            categories = feature["categories"]
            unseen_rate = float((~labels.isin(categories)).mean())
            proportions = _categorical_proportions(labels, categories)
            unseen_status = status_for_unseen(unseen_rate, thresholds.unseen_red)
        psi = population_stability_index(np.asarray(feature["proportions"]), proportions)
        psi_status = status_for_signal(psi, thresholds.psi_amber, thresholds.psi_red)
        feature_results[column] = {
            "psi": psi,
            "psi_status": psi_status,
            "missing_rate": missing_rate,
            "missing_rate_delta": missing_delta,
            "missing_status": missing_status,
            "unseen_category_rate": unseen_rate,
            "unseen_status": unseen_status,
        }
        statuses.extend([psi_status, missing_status, unseen_status])

    score_proportions = _histogram_proportions(scores, reference["score"]["bin_edges"])
    score_psi = population_stability_index(
        np.asarray(reference["score"]["proportions"]), score_proportions
    )
    score_status = status_for_signal(score_psi, thresholds.psi_amber, thresholds.psi_red)
    approval_rate = float((scores < reference["decision_threshold"]).mean())
    approval_delta = abs(approval_rate - float(reference["approval_rate"]))
    approval_status = status_for_signal(
        approval_delta, thresholds.approval_delta_amber, thresholds.approval_delta_red
    )
    statuses.extend([score_status, approval_status])

    performance = None
    if y_true is not None:
        y_true = np.asarray(y_true)
        if len(y_true) != len(X):
            raise ValueError("targets must match the batch feature length")
        performance = _metric_dict(y_true, scores)
        reference_performance = reference.get("performance")
        if reference_performance is None:
            performance.update({"auc_status": "not_comparable", "brier_status": "not_comparable"})
        else:
            auc_drop = max(0.0, float(reference_performance["auc"]) - performance["auc"])
            brier_increase = max(0.0, performance["brier"] - float(reference_performance["brier"]))
            performance.update(
                {
                    "auc_drop": auc_drop,
                    "auc_status": status_for_signal(
                        auc_drop, thresholds.auc_drop_amber, thresholds.auc_drop_red
                    ),
                    "brier_increase": brier_increase,
                    "brier_status": status_for_signal(
                        brier_increase,
                        thresholds.brier_increase_amber,
                        thresholds.brier_increase_red,
                    ),
                }
            )
            statuses.extend([performance["auc_status"], performance["brier_status"]])

    return {
        "model_version": reference["model_version"],
        "row_count": len(X),
        "overall_status": _maximum_status(statuses),
        "score": {"psi": score_psi, "status": score_status, "mean_pd": float(scores.mean())},
        "approval": {
            "rate": approval_rate,
            "absolute_delta": approval_delta,
            "status": approval_status,
        },
        "features": feature_results,
        "performance": performance,
    }
