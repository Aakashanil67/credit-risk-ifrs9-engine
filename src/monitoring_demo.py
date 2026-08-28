"""Deterministic replay and stress demonstration for public-demo monitoring controls."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.artifacts import load_artifact_bundle
from src.config import (
    FIGURES_DIR,
    MONITORING_SAMPLE_ROWS,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COL,
    model_bundle_dir,
)
from src.data_loader import load_application_data
from src.features import build_lgbm_features
from src.model_profiles import ModelProfile
from src.monitoring import (
    build_monitoring_reference,
    evaluate_monitoring_batch,
    save_monitoring_reference,
)
from src.preprocessing import split_data

_STRESS_SETTINGS = {
    "baseline_replay": {
        "income_multiplier": 1.0,
        "credit_multiplier": 1.0,
        "occupation_missing_rate": 0.0,
    },
    "mild_shift": {
        "income_multiplier": 0.90,
        "credit_multiplier": 1.05,
        "occupation_missing_rate": 0.10,
    },
    "severe_shift": {
        "income_multiplier": 0.70,
        "credit_multiplier": 1.20,
        "occupation_missing_rate": 0.25,
    },
}


def apply_stress(batch: pd.DataFrame, severity: str, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return an unchanged or stressed copy using only the documented replay transformations."""
    if severity not in _STRESS_SETTINGS:
        raise ValueError(f"unsupported monitoring severity: {severity}")
    required = {"AMT_INCOME_TOTAL", "AMT_CREDIT", "OCCUPATION_TYPE"}
    missing = sorted(required.difference(batch.columns))
    if missing:
        raise ValueError(f"batch is missing stress fields: {missing}")
    settings = _STRESS_SETTINGS[severity]
    result = batch.copy()
    result["AMT_INCOME_TOTAL"] *= settings["income_multiplier"]
    result["AMT_CREDIT"] *= settings["credit_multiplier"]
    eligible = result.index[result["OCCUPATION_TYPE"].notna()].to_numpy()
    count = min(round(settings["occupation_missing_rate"] * len(result)), len(eligible))
    if count:
        chosen = np.random.default_rng(seed).choice(eligible, size=count, replace=False)
        result.loc[chosen, "OCCUPATION_TYPE"] = np.nan
    return result


def _align_categories(X: pd.DataFrame, category_dtypes: dict) -> pd.DataFrame:
    aligned = X.copy()
    for column, dtype in category_dtypes.items():
        aligned[column] = aligned[column].astype(dtype)
    return aligned


def _score_raw_frame(
    raw: pd.DataFrame, bundle, profile: ModelProfile
) -> tuple[pd.DataFrame, np.ndarray]:
    features = _align_categories(build_lgbm_features(raw, profile=profile), bundle.category_dtypes)
    return features, bundle.model.predict_proba(features)[:, 1]


def _batch_record(name: str, settings: dict, monitoring: dict) -> dict:
    return {"name": name, "transformations": settings, "monitoring": monitoring}


def _top_features(batch: dict, limit: int = 5) -> list[tuple[str, dict]]:
    return sorted(
        batch["monitoring"]["features"].items(), key=lambda item: item[1]["psi"], reverse=True
    )[:limit]


def render_monitoring_report(result: dict) -> str:
    """Render a clear simulation report without presenting stress data as observed production data."""
    lines = [
        "# Monitoring replay and stress demonstration",
        "",
        "This is simulated monitoring, built from a deterministic Home Credit test-fold replay and "
        "controlled feature stress. These batches are not production observations and their retained "
        "historical labels are not newly observed outcomes.",
        "",
    ]
    if not result.get("batches"):
        return "\n".join(lines)
    lines += [
        f"- Development reference rows: **{result['reference_rows']:,}**",
        f"- Replay rows per batch: **{result['replay_rows']:,}**",
        f"- Model version: **{result['model_version']}**",
        "",
        "| batch | overall status | score PSI | approval rate | AUC | Brier |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for batch in result["batches"]:
        monitoring = batch["monitoring"]
        performance = monitoring["performance"]
        lines.append(
            f"| {batch['name']} | {monitoring['overall_status']} | {monitoring['score']['psi']:.3f} | "
            f"{monitoring['approval']['rate']:.2%} | {performance['auc']:.4f} | {performance['brier']:.4f} |"
        )
    lines += [
        "",
        "## Largest feature PSI values",
        "",
    ]
    for batch in result["batches"]:
        lines += [f"### {batch['name']}", "", "| feature | PSI | status |", "|---|---:|---|"]
        for feature, values in _top_features(batch):
            lines.append(f"| {feature} | {values['psi']:.3f} | {values['psi_status']} |")
        lines.append("")
    lines += [
        "## Transformations",
        "",
    ]
    for batch in result["batches"]:
        lines.append(
            f"- **{batch['name']}:** `{json.dumps(batch['transformations'], sort_keys=True)}`"
        )
    lines += [
        "",
        "## Demonstration thresholds",
        "",
        "| Signal | Amber | Red |",
        "|---|---:|---:|",
        "| Feature or score PSI | >= 0.10 | >= 0.25 |",
        "| Missing-rate movement | >= 5 percentage points | >= 10 percentage points |",
        "| Unseen category rate | > 0% | >= 1% |",
        "| Approval-rate movement | >= 5 percentage points | >= 10 percentage points |",
        "| AUC decrease with labels | >= 0.02 | >= 0.05 |",
        "| Brier increase with labels | >= 0.005 | >= 0.010 |",
        "",
        "Static competition outcomes cannot demonstrate delayed-label monitoring, causal macroeconomic "
        "stress, source-system change, or real production drift. The thresholds are illustrative controls, "
        "not universal lending-policy limits.",
        "",
    ]
    return "\n".join(lines)


def _plot_psi(result: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in result["batches"]:
        for feature, values in _top_features(batch):
            rows.append({"batch": batch["name"], "feature": feature, "psi": values["psi"]})
    chart = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    if not chart.empty:
        labels = [f"{row.batch}: {row.feature}" for row in chart.itertuples(index=False)]
        ax.barh(labels, chart["psi"], color="#2f6f9f")
    ax.axvline(0.10, color="#d18f00", linestyle="--", label="amber 0.10")
    ax.axvline(0.25, color="#b23a48", linestyle="--", label="red 0.25")
    ax.set_xlabel("Population stability index")
    ax.set_title("Largest monitored feature shifts by replay batch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monitoring_psi.png", dpi=150)
    plt.close(fig)


def run_monitoring_demo() -> dict:
    """Build reference distributions and score deterministic replay/stress batches."""
    profile = ModelProfile.PUBLIC_DEMO
    bundle = load_artifact_bundle(model_bundle_dir(profile.value))
    train, validation, test = split_data(load_application_data(), seed=RANDOM_SEED)
    development = pd.concat([train, validation], ignore_index=True)
    development_X, development_scores = _score_raw_frame(development, bundle, profile)
    reference = build_monitoring_reference(
        development_X,
        development_scores,
        threshold=(0.08 / 0.57),
        model_version=bundle.metadata["model_version"],
        y_true=development[TARGET_COL].to_numpy(),
    )
    save_monitoring_reference(
        reference, model_bundle_dir(profile.value) / "monitoring_reference.json"
    )
    replay = test.sample(
        n=min(MONITORING_SAMPLE_ROWS, len(test)), random_state=RANDOM_SEED
    ).reset_index(drop=True)
    raw_features = replay.drop(columns=[TARGET_COL])
    batches = []
    for name, settings in _STRESS_SETTINGS.items():
        stressed = apply_stress(raw_features, name, seed=RANDOM_SEED)
        current_X, scores = _score_raw_frame(stressed, bundle, profile)
        monitoring = evaluate_monitoring_batch(
            reference, current_X, scores, y_true=replay[TARGET_COL].to_numpy()
        )
        batches.append(_batch_record(name, settings, monitoring))
    return {
        "model_version": bundle.metadata["model_version"],
        "reference_rows": len(development),
        "replay_rows": len(replay),
        "seed": RANDOM_SEED,
        "batches": batches,
    }


def write_outputs(result: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "monitoring_demo.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (REPORTS_DIR / "monitoring_demo.md").write_text(
        render_monitoring_report(result), encoding="utf-8"
    )
    _plot_psi(result)


def main() -> None:
    result = run_monitoring_demo()
    write_outputs(result)
    print(f"wrote monitoring demo for {result['replay_rows']:,} replay rows")


if __name__ == "__main__":
    main()
