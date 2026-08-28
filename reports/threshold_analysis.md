# Decision threshold

The service uses an illustrative expected-value threshold, not the population default rate: applications at or above the threshold are declined. With a 12% performing margin, 2% operating cost, 2% capital cost, and 45% LGD, the break-even PD is 0.140351.

## Untouched test-fold operating view

- Approval rate: **89.25%**
- Default recall among declined applications: **25.86%**
- Observed default precision among declined applications: **19.42%**
- Confusion matrix (actual default positive): TP 1,284, FP 5,328, TN 51,210, FN 3,681.
- Calibration intercept: **0.1336**; calibration slope: **1.0560**.

## Test-fold model uncertainty

Intervals are 95% stratified bootstrap intervals from the untouched historical test fold; they quantify sampling uncertainty, not future portfolio performance.

| metric | estimate | 95% stratified bootstrap interval |
|---|---:|---:|
| auc | 0.6774 | 0.6697 to 0.6856 |
| gini | 0.3547 | 0.3393 to 0.3711 |
| ks | 0.2611 | 0.2477 to 0.2764 |
| brier | 0.0716 | 0.0713 to 0.0719 |
| pr auc | 0.1612 | 0.1549 to 0.1685 |
| log loss | 0.2652 | 0.2638 to 0.2665 |

The numbers describe one historical competition split and are not a lending policy, an approval recommendation, or evidence of profitability. A lender would estimate product pricing, capital, LGD, prepayment, collections, and constraints from its own portfolio before approving any threshold.
