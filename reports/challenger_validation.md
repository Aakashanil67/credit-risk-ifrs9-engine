# Development-only challenger validation

This development-only study compares candidates using five-fold out-of-fold predictions on train plus validation data. The test fold was not used for candidate selection or this challenger report. It is evidence for a later review, not a model promotion.

- Incumbent model version: **1.2.0**
- Development rows: **246,008**
- Folds / seed / bootstrap samples: **5 / 42 / 1,000**
- Paired-bootstrap sample rows: **20000**

| candidate | AUC | Gini | Brier | Log loss | gate | result |
|---|---:|---:|---:|---:|---|---|
| base_raw | 0.6717 | 0.3434 | 0.0718 | 0.2662 | incumbent | retained |
| base_sigmoid | 0.6717 | 0.3433 | 0.0718 | 0.2662 | calibration | fail |
| base_isotonic | 0.6707 | 0.3415 | 0.0718 | 0.2665 | calibration | fail |
| derived_raw | 0.6736 | 0.3473 | 0.0718 | 0.2658 | derived | fail |
| derived_sigmoid | 0.6736 | 0.3472 | 0.0718 | 0.2659 | calibration | fail |
| derived_isotonic | 0.6727 | 0.3453 | 0.0718 | 0.2662 | calibration | fail |

## Predeclared gate results

| candidate | result |
|---|---|
| base_sigmoid | Brier improvement is smaller than 0.0005. Brier interval does not remain below zero. Log loss worsens. |
| base_isotonic | Brier improvement is smaller than 0.0005. Brier interval does not remain below zero. Log loss worsens. |
| derived_raw | AUC improvement is smaller than 0.003. AUC interval crosses zero. |
| derived_sigmoid | Brier improvement is smaller than 0.0005. Brier interval does not remain below zero. Log loss worsens. Derived-feature parent did not pass its acceptance gate. |
| derived_isotonic | Brier improvement is smaller than 0.0005. Brier interval does not remain below zero. Log loss worsens. Derived-feature parent did not pass its acceptance gate. |

## Outcome

No candidate passed every predeclared gate, so the incumbent remains preferred.

Candidate gate intervals come from stratified m-out-of-n paired bootstrap samples. Each point estimate still uses the full development sample. This is not nested cross-validation. The incumbent's frozen parameters and tree count come from the earlier v1.2 development process. The out-of-fold (OOF) protocol stops an estimator or calibrator from scoring a row it was fitted on. Even so, this report is development evidence, not an independent validation.
