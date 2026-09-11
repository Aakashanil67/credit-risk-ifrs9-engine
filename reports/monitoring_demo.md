# Monitoring replay and stress demonstration

This is simulated monitoring, built from a deterministic Home Credit test-fold replay and controlled feature stress. These batches are not production observations and their retained historical labels are not newly observed outcomes.

The monitoring reference is an out-of-sample monitoring reference window, disjoint from the replay window.

- Out-of-sample reference rows: **51,503**
- Replay rows per batch: **10,000**
- Model version: **1.2.0**

| batch | overall status | score PSI | approval rate | AUC | Brier |
|---|---|---:|---:|---:|---:|
| baseline_replay | green | 0.001 | 89.50% | 0.6892 | 0.0713 |
| mild_shift | red | 0.005 | 88.54% | 0.6883 | 0.0713 |
| severe_shift | red | 0.092 | 84.83% | 0.6821 | 0.0715 |

## Largest feature PSI values

### baseline_replay

| feature | PSI | status |
|---|---:|---|
| OCCUPATION_TYPE | 0.002 | green |
| AMT_ANNUITY | 0.001 | green |
| NAME_INCOME_TYPE | 0.001 | green |
| FLAG_OWN_REALTY | 0.001 | green |
| AMT_GOODS_PRICE | 0.001 | green |

### mild_shift

| feature | PSI | status |
|---|---:|---|
| AMT_INCOME_TOTAL | 0.203 | amber |
| OCCUPATION_TYPE | 0.048 | green |
| AMT_CREDIT | 0.013 | green |
| AMT_ANNUITY | 0.001 | green |
| NAME_INCOME_TYPE | 0.001 | green |

### severe_shift

| feature | PSI | status |
|---|---:|---|
| AMT_INCOME_TOTAL | 0.804 | red |
| OCCUPATION_TYPE | 0.266 | red |
| AMT_CREDIT | 0.132 | amber |
| AMT_ANNUITY | 0.001 | green |
| NAME_INCOME_TYPE | 0.001 | green |

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
