# Model card: public-demo PD model

## Intended use

This is an educational probability-of-default model for a consumer-loan application. The FastAPI
service and Streamlit dashboard make its input contract, explanation path, and illustrative
decision rule inspectable. It is not approved for lending, product pricing, collections,
accounting, or automated adverse action.

## Data and model contract

The model uses Kaggle's Home Credit Default Risk `application_train.csv`: 307,511 historical
applications with 122 raw columns and an 8.1% event rate. `TARGET=1` denotes payment difficulty
under the competition's definition; the public dataset does not disclose the exact delinquency-day
threshold. The data is not South African and should not be treated as a local portfolio.

The served contract has 15 application-time fields: contract type, age, employment tenure, income,
credit amount, annuity, goods price, car and property ownership, household counts, education,
income type, family status, and occupation. It excludes gender, external credit-bureau variables,
organisation type, regional density, and car age. The dashboard exposes every required field and
its categorical choices are constrained to the fitted category levels.

The endpoint rejects values outside its public contract before scoring. Income is limited to
5,000,000, credit and goods price to 4,050,000, and annuity to 300,000 dataset monetary units;
these limits prevent source-data outliers and malformed requests from producing misleading reason
codes. It also allows at most 20 predictions per IP address per minute on the single-instance demo.

Data is split 60/20/20 with stratification and seed 42. Five-fold cross-validation on the training
fold selects hyperparameters. Early stopping on validation chooses the tree count. The selected
model is refit on the combined train and validation folds, and the test fold is used once for the
final performance estimate. The fitted bundle records its data hash, feature order, category
levels, package versions, selection settings, and test metrics in `models/public_demo/metadata.json`.

## Performance

| Metric | Logistic baseline | LightGBM |
|---|---:|---:|
| AUC | 0.6563 | 0.6774 |
| Gini | 0.3127 | 0.3547 |
| KS | 0.2319 | 0.2611 |
| Brier score | 0.0723 | 0.0716 |
| PR-AUC | 0.1444 | 0.1612 |
| Log loss | 0.2691 | 0.2652 |

The logistic model uses the same 15 fields and the same train/validation and test partitions as
LightGBM. The result shows a modest discrimination and probability-error improvement on this one
historical split. It does not establish out-of-time stability, portability, or commercial value.

The full test-fold uncertainty results are in the [public-demo audit](public_demo_audit.json) and
[validation report](validation_report.md). Intervals use 1,000 deterministic stratified bootstrap
samples. They quantify uncertainty on this historical split, not future portfolio performance.

At the illustrative threshold of 0.140351, the test-fold approval rate is 89.25%, default recall
among declined applications is 25.86%, and observed default precision among declines is 19.42%.
The calibration intercept is 0.1336 and slope is 1.0560. These are diagnostics, not operating
targets; the full matrix and assumptions are in [threshold analysis](threshold_analysis.md).

## Explanations and loss estimates

Each response includes the three largest local SHAP contributions. They describe how fitted model
features moved a score relative to the model baseline; they are not causal findings or legally
sufficient adverse-action reasons. Raw day-count features are translated into years and amounts
into dataset monetary units before being shown.

Global SHAP plots and the calibration curve use a 3,000-row sample and the full untouched test
fold respectively. The final model is trained on train plus validation data, so using validation
rows for those published artefacts would have made them in-sample diagnostics.

The dashboard displays an illustrative 12-month loss estimate: `PD × 45% LGD × requested credit`.
It is not an IFRS 9 provision. The separate ECL report adds stage assignment, survival-weighted
monthly hazards, scenario weighting, and discounting on fixed accounts. It still lacks observed
recoveries, contractual cash flows, and account-level PD history.

## Fairness, governance, and monitoring

Gender is kept only for an offline test-fold diagnostic and is absent from the served inputs. The
fairness audit shows a gender approval-rate difference and differing group error rates under the
illustrative threshold. Removing a direct feature does not remove proxy risk. This analysis is not
a disparate-impact assessment, legal review, or production monitoring programme.

A real deployment would need local outcome data, legal and policy review, independent validation,
data-quality controls, reason-code governance, drift monitoring, retraining approval, and human
oversight. None of those controls can be inferred from the Home Credit competition data.

## Lifecycle controls

The v1.3 service release adds development-only challenger evaluation, an aggregate monitoring
reference, privacy-safe request telemetry, and a consolidated [validation report](validation_report.md).
The challenger study found that no candidate passed every predeclared gate, so the fitted v1.2.0
artifact remains deployed unless a separate promotion review approves a different bundle.

The [monitoring demonstration](monitoring_demo.md) and [monitoring runbook](monitoring_runbook.md)
use a disjoint held-out test reference window, deterministic replay and controlled stress batches.
They are simulations, not real-time or production monitoring. The monitoring reference retains
aggregate distributions only.
