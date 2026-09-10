# Consolidated validation report: public-demo PD model

## 1. Scope and validation status

This is an educational control pack for a historical Home Credit demonstration. It is not an independent validation, a production approval, or evidence that the model is suitable for a lender's portfolio.

## 2. Model and service identification

- Model version: **1.2.0**
- Service release: **1.3.0**
- The fitted v1.2.0 model bundle remains unchanged by this lifecycle release.

## 3. Data and split protocol

The work uses the historical Kaggle Home Credit application dataset with a deterministic 60/20/20 stratified split (seed 42). Candidate selection uses development-only out-of-fold predictions; the frozen test fold is not used to choose or promote a challenger.

- Train rows: **184506**
- Validation rows: **61502**
- Test rows: **61503**

## 4. Incumbent test evidence and intervals

The following intervals use 1,000 deterministic stratified bootstrap resamples of the untouched historical test fold.

| metric | estimate | 95% interval |
|---|---:|---:|
| auc | 0.6774 | 0.6697 to 0.6856 |
| brier | 0.0716 | 0.0713 to 0.0719 |
| gini | 0.3547 | 0.3393 to 0.3711 |
| ks | 0.2611 | 0.2477 to 0.2764 |
| log loss | 0.2652 | 0.2638 to 0.2665 |
| pr auc | 0.1612 | 0.1549 to 0.1685 |

Metadata records test AUC as **0.677370713463237**.

## 5. Development-only challenger evidence and gate outcome

- Data scope: **development_oof_only**
- Candidates evaluated: **5**; rejected: **5**.
- No candidate passed every predeclared gate; the incumbent remains preferred.

## 6. Fairness diagnostic summary and limits

The offline gender diagnostic estimates a female-minus-male approval-rate difference of **8.06%** (95% interval 7.49% to 8.64%). It is a screening diagnostic only: it does not establish fairness, disparate impact, causality, or legal compliance.

## 7. Monitoring reference and stress results

These are simulated monitoring replays and controlled stresses, not production observations.
- Reference scope: **frozen_test_reference_window**
- Reference rows: **51,503**
- Replay rows: **10,000**
The reference and replay windows are disjoint and both are out of sample relative to model fitting.

| batch | status |
|---|---|
| baseline_replay | green |
| mild_shift | red |
| severe_shift | red |

## 8. Decision economics and threshold sensitivity limits

The illustrative expected-value threshold is **0.140351**. It is not a lending policy or proof of profitability; a lender would need local pricing, LGD, capital, collections, and policy constraints.

## 9. Model limitations

The dataset is historical competition data, not a local portfolio. It contains no out-of-time validation, realised production monitoring, recovery experience, or evidence of portability to South African or other lending contexts.

## 10. Acceptance matrix

| control | status | evidence or limitation |
|---|---|---|
| Reproducibility | met | Deterministic split, seed, versioned bundle, and generated reports. |
| Discrimination | partially met | Historical test-fold intervals are published; no out-of-time evidence. |
| Calibration | partially met | Historical Brier and calibration diagnostics are published. |
| Public input contract | met | The served 15-field schema is versioned and tested. |
| Explanations | partially met | Local SHAP explanations are diagnostic, not causal adverse-action reasons. |
| Fairness diagnostics | partially met | Offline group diagnostics exist; no legal or local assessment. |
| Monitoring design | partially met | Aggregate reference and simulations exist; no live feed. |
| Local data | not met | No lender-specific development or outcome data. |
| Out-of-time evidence | not met | No temporal holdout or portfolio performance study. |
| Independent validation | not met | This solo-project report is not an independent validation. |

## 11. Required production controls not demonstrated here

A real deployment would require approved data lineage, schema controls, local outcome monitoring, independent validation, policy and legal review, secure audit retention, incident response, controlled retraining, and formal model-promotion governance.
