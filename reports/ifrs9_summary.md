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

## Calculation method

For Stages 1 and 2, annual PD is converted to a constant monthly hazard. Each month's loss uses the probability that the account has survived to that month and defaults during that month, multiplied by LGD and EAD, then discounted at the effective interest rate. Stage 1 is capped at 12 months; Stage 2 runs over the remaining term. The result is weighted across the stated upside, base, and downside scenarios (20% / 60% / 20%).

For Stage 3, the demonstration treats the account as already in default and calculates the first discounted cash shortfall (LGD × EAD), rather than applying another stream of default probabilities. A production Stage 3 model would project recoveries and costs from workout cash flows.

## What this does not model

The source data does not contain contractual amortisation schedules, account balances over time, observed transitions between stages, recoveries, forward-looking macroeconomic variables, or a lender's approved SICR policy. Replacing those assumptions is necessary before using this method for accounting or credit decisions.
