# IFRS 9 expected credit loss mechanics

This report demonstrates the project's ECL calculation on three fixed accounts. Amounts are expressed in **dataset monetary units**, because the Home Credit competition data does not identify a currency. It is not a portfolio provision or a claim about a lender's actual expected loss.

## Stage rules used in this demonstration

- **Stage 1:** 12-month ECL when there is no significant increase in credit risk (SICR).
- **Stage 2:** lifetime ECL when days past due are at least 30, or current PD is at least double origination PD and at least 5%.
- **Stage 3:** credit-impaired when days past due are at least 90 or the account is flagged credit-impaired.

## Scenario-weighted discounted examples

| account | stage | origination PD | current annual PD | EAD (monetary units) | remaining months | ECL (monetary units) | coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| Performing account | Stage 1 - performing | 3.0% | 4.0% | 100,000 | 48 | 1,796 | 1.80% |
| SICR account | Stage 2 - significant increase in credit risk | 3.0% | 8.0% | 100,000 | 48 | 10,889 | 10.89% |
| Credit-impaired account | Stage 3 - credit-impaired | 3.0% | 40.0% | 100,000 | 48 | 45,568 | 45.57% |

## Worked Stage 1 example

The performing account starts with annual PD of 4%, EAD of 100,000 and an effective interest rate of 12%. EAD is held constant because the source data has no contractual repayment schedule. The base scenario uses 45% LGD and carries 60% of the final result.

The base-scenario monthly calculation is shown in full. `Survival` is the probability that the account reaches the start of the month without default. `Marginal PD` is the probability of surviving to that month and defaulting during it.

| month | survival | marginal PD | discount factor | discounted loss before weighting |
|---:|---:|---:|---:|---:|
| 1 | 1.000000 | 0.003396 | 0.990600 | 151.39 |
| 2 | 0.996604 | 0.003385 | 0.981289 | 149.45 |
| 3 | 0.993219 | 0.003373 | 0.972065 | 147.55 |
| 4 | 0.989846 | 0.003362 | 0.962928 | 145.66 |
| 5 | 0.986485 | 0.003350 | 0.953877 | 143.80 |
| 6 | 0.983135 | 0.003339 | 0.944911 | 141.97 |
| 7 | 0.979796 | 0.003327 | 0.936029 | 140.16 |
| 8 | 0.976468 | 0.003316 | 0.927231 | 138.37 |
| 9 | 0.973152 | 0.003305 | 0.918515 | 136.60 |
| 10 | 0.969847 | 0.003294 | 0.909882 | 134.86 |
| 11 | 0.966554 | 0.003282 | 0.901329 | 133.14 |
| 12 | 0.963271 | 0.003271 | 0.892857 | 131.44 |

The three scenario totals reconcile to the Stage 1 result above:

| scenario | weight | adjusted annual PD | LGD | loss before weighting | weighted loss |
|---|---:|---:|---:|---:|---:|
| Upside | 20% | 3.20% | 40% | 1,204.80 | 240.96 |
| Base | 60% | 4.00% | 45% | 1,694.38 | 1,016.63 |
| Downside | 20% | 5.20% | 55% | 2,692.49 | 538.50 |
| **Total** | **100%** |  |  |  | **1,796.08** |

## Calculation method

For Stages 1 and 2, annual PD is converted to a constant monthly hazard. Each month's loss uses the probability that the account has survived to that month and defaults during that month, multiplied by LGD and EAD, then discounted at the effective interest rate. Stage 1 includes defaults arising in the next 12 months; Stage 2 runs over the remaining term. The result is weighted across the stated upside, base, and downside scenarios (20% / 60% / 20%).
Because the example has no contractual cash-flow schedule, it recognises the assumed loss at the default month. A production Stage 1 calculation would estimate lifetime cash shortfalls associated with defaults that can occur during the next 12 months.

For Stage 3, the demonstration treats the account as already in default and calculates the first discounted cash shortfall (LGD × EAD), rather than applying another stream of default probabilities. A production Stage 3 model would project recoveries and costs from workout cash flows.

## What this does not model

The 30- and 90-day arrears backstops, the two-times PD trigger and the 5% floor are policy assumptions for this example, not calibrated SICR rules. The source data does not contain contractual amortisation schedules, account balances over time, observed transitions between stages, recoveries, forward-looking macroeconomic variables, or a lender's approved SICR policy. Replacing those assumptions is necessary before using this method for accounting or credit decisions.
