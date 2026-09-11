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

- **base_sigmoid:** Brier improvement is smaller than 0.0005; Brier interval does not remain below zero; Log loss worsens
- **base_isotonic:** Brier improvement is smaller than 0.0005; Brier interval does not remain below zero; Log loss worsens
- **derived_raw:** AUC improvement is smaller than 0.003; AUC interval crosses zero
- **derived_sigmoid:** Brier improvement is smaller than 0.0005; Brier interval does not remain below zero; Log loss worsens; Derived-feature parent did not pass its acceptance gate
- **derived_isotonic:** Brier improvement is smaller than 0.0005; Brier interval does not remain below zero; Log loss worsens; Derived-feature parent did not pass its acceptance gate

## Outcome

No candidate passed every predeclared gate; the incumbent remains preferred.

The study uses stratified m-out-of-n paired bootstrap samples for candidate gate intervals; the full development sample is still used for each point estimate. The study is not nested cross-validation: the incumbent's frozen parameters and tree count come from the prior v1.2 development process. The OOF protocol prevents an estimator or calibrator from scoring a row it fitted, but this report is still development evidence rather than an independent validation.
