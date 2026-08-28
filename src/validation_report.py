"""Build a consolidated, evidence-linked validation report for the public demo."""

import json
from typing import Any

from src.config import REPORTS_DIR, SERVICE_VERSION, model_bundle_dir


def _require(source: dict[str, Any], key: str, source_name: str) -> Any:
    if key not in source:
        raise ValueError(f"{source_name} is missing required key: {key}")
    return source[key]


def _batch_status(batch: dict[str, Any]) -> str:
    monitoring = batch.get("monitoring", batch)
    return str(_require(monitoring, "overall_status", "monitoring batch"))


def _metric_interval_rows(intervals: dict[str, dict[str, float]]) -> list[str]:
    return [
        f"| {name.replace('_', ' ')} | {values['estimate']:.4f} | "
        f"{values['lower']:.4f} to {values['upper']:.4f} |"
        for name, values in intervals.items()
    ]


def build_validation_report(
    metadata: dict[str, Any],
    audit: dict[str, Any],
    challenger: dict[str, Any],
    monitoring: dict[str, Any],
) -> str:
    """Render an evidence-led report without converting demo outputs into production claims."""
    model_version = _require(metadata, "model_version", "metadata")
    test_metrics = _require(metadata, "test_metrics", "metadata")
    metric_intervals = _require(audit, "metric_intervals", "audit")
    gender_gap = _require(audit, "gender_approval_gap", "audit")
    data_scope = _require(challenger, "data_scope", "challenger")
    nomination = challenger.get("nomination")
    candidates = _require(challenger, "candidates", "challenger")
    batches = _require(monitoring, "batches", "monitoring")
    if not batches:
        raise ValueError("monitoring batches must not be empty")

    batch_rows = [f"| {batch['name']} | {_batch_status(batch)} |" for batch in batches]
    challenger_result = (
        f"Candidate **{nomination}** was nominated for a separate promotion review."
        if nomination
        else "No candidate passed every predeclared gate; the incumbent remains preferred."
    )
    rejected = sum(not candidate.get("accepted", False) for candidate in candidates)
    split_rows = metadata.get("split_rows", {})
    threshold = audit.get("threshold_metrics", {}).get("threshold")
    threshold_text = f"**{threshold:.6f}**" if threshold is not None else "not recorded"

    lines = [
        "# Consolidated validation report: public-demo PD model",
        "",
        "## 1. Scope and validation status",
        "",
        "This is an educational control pack for a historical Home Credit demonstration. It is "
        "not an independent validation, a production approval, or evidence that the model is "
        "suitable for a lender's portfolio.",
        "",
        "## 2. Model and service identification",
        "",
        f"- Model version: **{model_version}**",
        f"- Service release: **{SERVICE_VERSION}**",
        "- The fitted v1.2.0 model bundle remains unchanged by this lifecycle release.",
        "",
        "## 3. Data and split protocol",
        "",
        "The work uses the historical Kaggle Home Credit application dataset with a deterministic "
        "60/20/20 stratified split (seed 42). Candidate selection uses development-only out-of-fold "
        "predictions; the frozen test fold is not used to choose or promote a challenger.",
        "",
        f"- Train rows: **{split_rows.get('train', 'not recorded')}**",
        f"- Validation rows: **{split_rows.get('validation', 'not recorded')}**",
        f"- Test rows: **{split_rows.get('test', audit.get('test_rows', 'not recorded'))}**",
        "",
        "## 4. Incumbent test evidence and intervals",
        "",
        "The following intervals use 1,000 deterministic stratified bootstrap resamples of the "
        "untouched historical test fold.",
        "",
        "| metric | estimate | 95% interval |",
        "|---|---:|---:|",
        *_metric_interval_rows(metric_intervals),
        "",
        f"Metadata records test AUC as **{test_metrics.get('AUC', 'not recorded')}**.",
        "",
        "## 5. Development-only challenger evidence and gate outcome",
        "",
        f"- Data scope: **{data_scope}**",
        f"- Candidates evaluated: **{len(candidates)}**; rejected: **{rejected}**.",
        f"- {challenger_result}",
        "",
        "## 6. Fairness diagnostic summary and limits",
        "",
        "The offline gender diagnostic estimates a female-minus-male approval-rate difference of "
        f"**{gender_gap['estimate']:.2%}** (95% interval {gender_gap['lower']:.2%} to "
        f"{gender_gap['upper']:.2%}). It is a screening diagnostic only: it does not establish "
        "fairness, disparate impact, causality, or legal compliance.",
        "",
        "## 7. Monitoring reference and stress results",
        "",
        "These are simulated monitoring replays and controlled stresses, not production observations.",
        "",
        "| batch | status |",
        "|---|---|",
        *batch_rows,
        "",
        "## 8. Decision economics and threshold sensitivity limits",
        "",
        f"The illustrative expected-value threshold is {threshold_text}. It is not a lending policy "
        "or proof of profitability; "
        "a lender would need local pricing, LGD, capital, collections, and policy constraints.",
        "",
        "## 9. Model limitations",
        "",
        "The dataset is historical competition data, not a local portfolio. It contains no "
        "out-of-time validation, realised production monitoring, recovery experience, or evidence "
        "of portability to South African or other lending contexts.",
        "",
        "## 10. Acceptance matrix",
        "",
        "| control | status | evidence or limitation |",
        "|---|---|---|",
        "| Reproducibility | met | Deterministic split, seed, versioned bundle, and generated reports. |",
        "| Discrimination | partially met | Historical test-fold intervals are published; no out-of-time evidence. |",
        "| Calibration | partially met | Historical Brier and calibration diagnostics are published. |",
        "| Public input contract | met | The served 15-field schema is versioned and tested. |",
        "| Explanations | partially met | Local SHAP explanations are diagnostic, not causal adverse-action reasons. |",
        "| Fairness diagnostics | partially met | Offline group diagnostics exist; no legal or local assessment. |",
        "| Monitoring design | partially met | Aggregate reference and simulations exist; no live feed. |",
        "| Local data | not met | No lender-specific development or outcome data. |",
        "| Out-of-time evidence | not met | No temporal holdout or portfolio performance study. |",
        "| Independent validation | not met | This solo-project report is not an independent validation. |",
        "",
        "## 11. Required production controls not demonstrated here",
        "",
        "A real deployment would require approved data lineage, schema controls, local outcome "
        "monitoring, independent validation, policy and legal review, secure audit retention, "
        "incident response, controlled retraining, and formal model-promotion governance.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Read generated aggregate artifacts and write the validation report."""
    metadata_path = model_bundle_dir("public_demo") / "metadata.json"
    sources = {
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
        "audit": json.loads((REPORTS_DIR / "public_demo_audit.json").read_text(encoding="utf-8")),
        "challenger": json.loads(
            (REPORTS_DIR / "challenger_validation.json").read_text(encoding="utf-8")
        ),
        "monitoring": json.loads(
            (REPORTS_DIR / "monitoring_demo.json").read_text(encoding="utf-8")
        ),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "validation_report.md"
    out_path.write_text(build_validation_report(**sources), encoding="utf-8")
    print(f"wrote {out_path.name}")


if __name__ == "__main__":
    main()
