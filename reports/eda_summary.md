# EDA summary

## Why accuracy is the wrong metric

8.1% of applicants had a recorded payment difficulty event. The competition does not disclose the exact delinquency threshold, so this project does not treat the target as a legally defined default. A model that predicts 'repaid' for every single applicant hits 91.9% accuracy while identifying none of those events. AUC, precision-recall, and the KS statistic (computed later against LightGBM) all separate the classes; accuracy collapses them.

## Five findings

1. **Missingness is structural, not random.** 67 of 122 columns have missing values, and the worst (COMMONAREA_MEDI) is missing 69.87% of the time. These are almost all building/apartment features that only exist for applicants who reported living in an apartment complex, so missing is informative rather than a data-entry gap. That argues against dropping them.

2. **`DAYS_EMPLOYED` has a sentinel-value bug.** 18.0% of rows carry the value 365243 (1,000 years), which is Home Credit's placeholder for 'not currently employed', not a real employment tenure. Left untreated it wrecks any model that uses employment length linearly.

3. **The three `EXT_SOURCE_*` columns dominate the correlation table.** EXT_SOURCE_3 correlates -0.179 with `TARGET`, more than any other raw feature in this table. They are external credit-bureau scores. The deployed public-demo model deliberately excludes them because an applicant cannot supply them.

4. **Income and credit amount are heavily right-skewed.** A handful of applicants report incomes in the tens of millions; the histograms clip at the 99th percentile so the bulk of the distribution is visible at all. The tree model uses the raw values; the logistic baseline standardises its numeric inputs.

5. **`DAYS_BIRTH` correlates 0.078 with `TARGET`.** Younger applicants default more often, which is consistent with having less credit history to underwrite against. It is not a causal age effect and shouldn't be read as one.

## Top correlations with TARGET

| feature | correlation |
|---|---|
| EXT_SOURCE_3 | -0.179 |
| EXT_SOURCE_2 | -0.160 |
| EXT_SOURCE_1 | -0.155 |
| DAYS_BIRTH | 0.078 |
| REGION_RATING_CLIENT_W_CITY | 0.061 |
