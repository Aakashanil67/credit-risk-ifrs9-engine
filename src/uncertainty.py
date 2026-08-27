"""Deterministic uncertainty estimates for binary-model metrics."""

from dataclasses import dataclass, fields

import numpy as np

from src.evaluation import binary_metrics


@dataclass(frozen=True)
class MetricInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_bootstrap: int


def _validate_inputs(y_true: np.ndarray, *scores: np.ndarray) -> None:
    if len(y_true) == 0 or any(len(score) != len(y_true) for score in scores):
        raise ValueError("targets and scores must have the same non-zero length")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("targets must contain both target classes")
    if any((~np.isfinite(score)).any() or ((score < 0) | (score > 1)).any() for score in scores):
        raise ValueError("scores must be finite probabilities between zero and one")


def _validate_bootstrap_options(n_bootstrap: int, confidence_level: float) -> None:
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between zero and one")


def _stratified_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sampled = [
        rng.choice(
            np.flatnonzero(y_true == target), size=int((y_true == target).sum()), replace=True
        )
        for target in (0, 1)
    ]
    return rng.permutation(np.concatenate(sampled))


def _metric_mapping(y_true: np.ndarray, pd_score: np.ndarray) -> dict[str, float]:
    metrics = binary_metrics(y_true, pd_score)
    return {field.name: float(getattr(metrics, field.name)) for field in fields(metrics)}


def _intervals(
    estimate: dict[str, float],
    samples: dict[str, list[float]],
    n_bootstrap: int,
    confidence_level: float,
) -> dict[str, MetricInterval]:
    alpha = (1 - confidence_level) / 2
    return {
        name: MetricInterval(
            estimate=value,
            lower=float(np.quantile(samples[name], alpha)),
            upper=float(np.quantile(samples[name], 1 - alpha)),
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
        )
        for name, value in estimate.items()
    }


def bootstrap_binary_metric_intervals(
    y_true: np.ndarray,
    pd_score: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, MetricInterval]:
    """Return deterministic stratified bootstrap intervals for binary-model metrics."""
    y_true = np.asarray(y_true)
    pd_score = np.asarray(pd_score, dtype=float)
    _validate_inputs(y_true, pd_score)
    _validate_bootstrap_options(n_bootstrap, confidence_level)

    estimate = _metric_mapping(y_true, pd_score)
    samples = {name: [] for name in estimate}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        index = _stratified_indices(y_true, rng)
        for name, value in _metric_mapping(y_true[index], pd_score[index]).items():
            samples[name].append(value)
    return _intervals(estimate, samples, n_bootstrap, confidence_level)


def paired_bootstrap_metric_deltas(
    y_true: np.ndarray,
    incumbent_pd: np.ndarray,
    challenger_pd: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, MetricInterval]:
    """Return paired challenger-minus-incumbent metric intervals on shared resamples."""
    y_true = np.asarray(y_true)
    incumbent_pd = np.asarray(incumbent_pd, dtype=float)
    challenger_pd = np.asarray(challenger_pd, dtype=float)
    _validate_inputs(y_true, incumbent_pd, challenger_pd)
    _validate_bootstrap_options(n_bootstrap, confidence_level)

    incumbent = _metric_mapping(y_true, incumbent_pd)
    challenger = _metric_mapping(y_true, challenger_pd)
    estimate = {name: challenger[name] - incumbent[name] for name in incumbent}
    samples = {name: [] for name in estimate}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        index = _stratified_indices(y_true, rng)
        incumbent_sample = _metric_mapping(y_true[index], incumbent_pd[index])
        challenger_sample = _metric_mapping(y_true[index], challenger_pd[index])
        for name in samples:
            samples[name].append(challenger_sample[name] - incumbent_sample[name])
    return _intervals(estimate, samples, n_bootstrap, confidence_level)


def bootstrap_group_rate_difference(
    approved: np.ndarray,
    groups: np.ndarray,
    group_a: str,
    group_b: str,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> MetricInterval:
    """Estimate the group-a minus group-b approval-rate interval by within-group resampling."""
    approved = np.asarray(approved, dtype=bool)
    groups = np.asarray(groups)
    if len(approved) == 0 or len(groups) != len(approved):
        raise ValueError("approved and groups must have the same non-zero length")
    _validate_bootstrap_options(n_bootstrap, confidence_level)
    a = approved[groups == group_a]
    b = approved[groups == group_b]
    if len(a) == 0 or len(b) == 0:
        raise ValueError("both requested groups must be present")

    estimate = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    samples = [
        float(
            rng.choice(a, size=len(a), replace=True).mean()
            - rng.choice(b, size=len(b), replace=True).mean()
        )
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - confidence_level) / 2
    return MetricInterval(
        estimate=estimate,
        lower=float(np.quantile(samples, alpha)),
        upper=float(np.quantile(samples, 1 - alpha)),
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )
