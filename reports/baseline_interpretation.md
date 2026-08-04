# Logistic regression baseline

Validation AUC: **0.7326**. Trained on 15 hand-picked, interpretable features — not the full 122-column table LightGBM gets later.

## Reading the coefficients

Each odds ratio is how much an applicant's odds of default multiply by for a one-unit increase in that feature, holding everything else fixed. An odds ratio above 1 raises default risk; below 1 lowers it.

Caveat: features aren't standardised, so 'one unit' means very different things across rows — a one-unit move in `REGION_POPULATION_RELATIVE` (range ~0-0.07) spans nearly its whole distribution, while a one-unit move in `AMT_CREDIT` (rand) is negligible, which is why its odds ratio rounds to 1.0000 even though the coefficient is real and significant. Compare effect sizes within a feature's own range, not across features by raw odds ratio.

## Top 5 statistically significant drivers

- **REGION_POPULATION_RELATIVE**: odds ratio 0.035 (p=3.52e-06). Each one-unit increase decreases the odds of default by 96.5%.
- **EXT_SOURCE_3**: odds ratio 0.071 (p=0.00e+00). Each one-unit increase decreases the odds of default by 92.9%.
- **EXT_SOURCE_2**: odds ratio 0.106 (p=0.00e+00). Each one-unit increase decreases the odds of default by 89.4%.
- **EXT_SOURCE_1**: odds ratio 0.299 (p=1.99e-69). Each one-unit increase decreases the odds of default by 70.1%.
- **is_male**: odds ratio 1.471 (p=3.06e-84). Each one-unit increase increases the odds of default by 47.1%.

## Full coefficient table

| feature | coef | odds ratio | p-value |
|---|---|---|---|
| EXT_SOURCE_2 | -2.2438 | 0.1060 | 0.00e+00 |
| EXT_SOURCE_3 | -2.6412 | 0.0713 | 0.00e+00 |
| AMT_GOODS_PRICE | -0.0000 | 1.0000 | 2.73e-99 |
| AMT_CREDIT | 0.0000 | 1.0000 | 2.01e-89 |
| is_male | 0.3857 | 1.4706 | 3.06e-84 |
| EXT_SOURCE_1 | -1.2064 | 0.2993 | 1.99e-69 |
| owns_car | -0.2813 | 0.7548 | 5.47e-42 |
| years_employed | -0.0265 | 0.9739 | 5.51e-42 |
| AMT_ANNUITY | 0.0000 | 1.0000 | 1.09e-28 |
| log_income | -0.1566 | 0.8550 | 4.42e-12 |
| age_years | -0.0043 | 0.9957 | 6.97e-07 |
| REGION_POPULATION_RELATIVE | -3.3553 | 0.0349 | 3.52e-06 |
| CNT_CHILDREN | 0.0563 | 1.0579 | 3.16e-02 |
| owns_realty | 0.0388 | 1.0395 | 4.52e-02 |
| CNT_FAM_MEMBERS | -0.0379 | 0.9628 | 6.51e-02 |
