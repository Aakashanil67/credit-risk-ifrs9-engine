# Fairness audit: application PD model

This audit evaluates the application-profile LightGBM model on the untouched test fold of 61,503
Home Credit applications. `CODE_GENDER` is retained only for this offline check; it is not a model
feature and the API does not accept it. The decision rule is the project's illustrative
expected-value threshold of 0.140351: an application is approved when its predicted PD falls below
that value.

## Gender groups

| group | n | observed default rate | mean predicted PD | Brier | ECE (10 bins) | approval rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F | 40,345 | 6.96% | 7.46% | 0.0630 | 0.0051 | 91.81% | 19.84% | 7.32% |
| M | 21,158 | 10.20% | 9.14% | 0.0881 | 0.0114 | 84.02% | 33.64% | 13.97% |

The gender approval-rate difference is 7.79 points. That difference is not caused by a
direct gender field, because the served model never receives one. It can still arise through
variables correlated with gender, observed outcomes, or both. The audit therefore shows a result to
investigate, not a finding that the model is fair. Calibration is weaker for the M group on this
split: its 10-bin calibration error is 0.0114, against 0.0051 for F.

## Age bands

| age band | n | observed default rate | mean predicted PD | approval rate | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|
| 18–24 | 2,426 | 12.28% | 12.15% | 69.25% | 0.0128 |
| 25–34 | 14,380 | 10.53% | 10.51% | 78.13% | 0.0026 |
| 35–44 | 16,833 | 8.50% | 8.40% | 88.44% | 0.0043 |
| 45–54 | 14,211 | 7.20% | 7.02% | 94.72% | 0.0019 |
| 55–64 | 12,050 | 5.34% | 5.50% | 99.17% | 0.0016 |
| 65+ | 1,603 | 3.49% | 4.06% | 100.00% | 0.0070 |

Age remains an explicit application feature because it is part of the original lending dataset and
has material predictive value. That is not a policy judgement. A lender would need legal review,
business justification and adverse-action controls before using it in a production decision.

This is a diagnostic on one historical dataset, not a disparate-impact assessment and not a
production fairness-monitoring programme. The next useful test would compare these figures against
local applicants and realised outcomes after a fixed monitoring period; this dataset cannot answer
whether the same differences persist in South Africa.
