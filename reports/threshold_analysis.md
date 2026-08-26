# Decision threshold

The service uses an illustrative expected-value threshold, not the population default rate: applications at or above the threshold are declined. With a 12% performing margin, 2% operating cost, 2% capital cost, and 45% LGD, the break-even PD is 0.140351.

## Untouched test-fold operating view

- Approval rate: **89.25%**
- Default recall among declined applications: **25.86%**
- Observed default precision among declined applications: **19.42%**
- Confusion matrix (actual default positive): TP 1,284, FP 5,328, TN 51,210, FN 3,681.
- Calibration intercept: **0.1336**; calibration slope: **1.0560**.

The numbers describe one historical competition split and are not a lending policy, an approval recommendation, or evidence of profitability. A lender would estimate product pricing, capital, LGD, prepayment, collections, and constraints from its own portfolio before approving any threshold.
