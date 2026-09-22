# Fairness audit: public-demo risk model

This audit evaluates the 15-field public-demo LightGBM model on its untouched Home Credit test fold. `CODE_GENDER` is retained only for this offline diagnostic; it is not a model feature and the API does not accept it. The illustrative decision threshold is 0.140351: applications below it are approved.

## Gender groups

| group | n | observed default rate | mean predicted PD | Brier | ECE (10 bins) | approval rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F | 40,345 | 6.96% | 7.43% | 0.0630 | 0.0048 | 92.02% | 20.20% | 7.06% |
| M | 21,158 | 10.20% | 9.18% | 0.0881 | 0.0102 | 83.96% | 33.23% | 14.08% |

## Age bands

| group | n | observed default rate | mean predicted PD | Brier | ECE (10 bins) | approval rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 18–24 | 2,426 | 12.28% | 12.27% | 0.1049 | 0.0050 | 69.91% | 44.30% | 28.10% |
| 25–34 | 14,380 | 10.53% | 10.55% | 0.0904 | 0.0062 | 77.86% | 42.14% | 19.79% |
| 35–44 | 16,833 | 8.50% | 8.37% | 0.0753 | 0.0027 | 88.80% | 24.67% | 9.95% |
| 45–54 | 14,211 | 7.20% | 6.98% | 0.0650 | 0.0021 | 94.91% | 14.47% | 4.37% |
| 55–64 | 12,050 | 5.34% | 5.50% | 0.0501 | 0.0017 | 99.27% | 2.02% | 0.66% |
| 65+ | 1,603 | 3.49% | 3.96% | 0.0336 | 0.0048 | 100.00% | 0.00% | 0.00% |

## Gender approval-rate difference

The estimate below is the female approval rate minus the male approval rate. Its 95% stratified bootstrap interval uses 1,000 deterministic resamples of this historical test fold.

- Estimate: **8.06 percentage points**
- 95% stratified bootstrap interval: **7.49 to 8.64 percentage points**

In this test fold, the gender groups differ in observed event mix, average model score, approval rate, and error rates. Those differences are descriptive. They do not identify whether the model, correlated input variables, the historical sample, or another factor caused the gap.

Differences in approval, error and calibration rates are signals for investigation, not proof of cause or fairness. Removing a direct gender feature does not rule out proxy effects. This is not a disparate-impact assessment, legal review, or production fairness-monitoring programme. A real assessment would require local applicants, realised outcomes, policy context, and governance review.
