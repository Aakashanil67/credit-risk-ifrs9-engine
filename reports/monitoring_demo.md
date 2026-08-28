# Monitoring replay and stress demonstration

This is simulated monitoring, built from a deterministic Home Credit test-fold replay and controlled feature stress. These batches are not production observations and their retained historical labels are not newly observed outcomes.

- Development reference rows: **246,008**
- Replay rows per batch: **10,000**
- Model version: **1.2.0**

| batch | overall status | score PSI | approval rate | AUC | Brier |
|---|---|---:|---:|---:|---:|
| baseline_replay | amber | 0.001 | 89.26% | 0.6743 | 0.0690 |
| mild_shift | red | 0.004 | 88.35% | 0.6732 | 0.0690 |
| severe_shift | red | 0.087 | 84.60% | 0.6703 | 0.0694 |

## Largest feature PSI values

### baseline_replay

| feature | PSI | status |
|---|---:|---|
| OCCUPATION_TYPE | 0.002 | green |
| AMT_INCOME_TOTAL | 0.002 | green |
| NAME_EDUCATION_TYPE | 0.001 | green |
| DAYS_BIRTH | 0.001 | green |
| AMT_CREDIT | 0.001 | green |

### mild_shift

| feature | PSI | status |
|---|---:|---|
| AMT_INCOME_TOTAL | 0.371 | red |
| OCCUPATION_TYPE | 0.051 | green |
| AMT_CREDIT | 0.019 | green |
| NAME_EDUCATION_TYPE | 0.001 | green |
| DAYS_BIRTH | 0.001 | green |

### severe_shift

| feature | PSI | status |
|---|---:|---|
| AMT_INCOME_TOTAL | 0.794 | red |
| OCCUPATION_TYPE | 0.274 | red |
| AMT_CREDIT | 0.163 | amber |
| NAME_EDUCATION_TYPE | 0.001 | green |
| DAYS_BIRTH | 0.001 | green |

## Transformations

- **baseline_replay:** `{"credit_multiplier": 1.0, "income_multiplier": 1.0, "occupation_missing_rate": 0.0}`
- **mild_shift:** `{"credit_multiplier": 1.05, "income_multiplier": 0.9, "occupation_missing_rate": 0.1}`
- **severe_shift:** `{"credit_multiplier": 1.2, "income_multiplier": 0.7, "occupation_missing_rate": 0.25}`

## Demonstration thresholds

| Signal | Amber | Red |
|---|---:|---:|
| Feature or score PSI | >= 0.10 | >= 0.25 |
| Missing-rate movement | >= 5 percentage points | >= 10 percentage points |
| Unseen category rate | > 0% | >= 1% |
| Approval-rate movement | >= 5 percentage points | >= 10 percentage points |
| AUC decrease with labels | >= 0.02 | >= 0.05 |
| Brier increase with labels | >= 0.005 | >= 0.010 |

Static competition outcomes cannot demonstrate delayed-label monitoring, causal macroeconomic stress, source-system change, or real production drift. The thresholds are illustrative controls, not universal lending-policy limits.
