# Baseline vs LightGBM

The deployed LightGBM model uses the application profile: 18 fields that can be supplied at
origination and no gender or bureau variables. Its parameters came from five-fold CV on the
training fold (`{'learning_rate': 0.05, 'num_leaves': 31}`); early stopping selected 166 trees on
validation data. The figures below are useful context, not a direct model contest: the logistic
baseline uses a richer, bureau-based feature set and is measured on validation data, while the
LightGBM figures are the single final estimate on the untouched test fold.

That distinction matters.

| metric | logistic baseline (validation) | application LightGBM (test) | difference |
|---|---|---|---|
| AUC | 0.7283 | 0.6787 | -0.0497 |
| Gini | 0.4567 | 0.3574 | -0.0993 |
| KS | 0.3390 | 0.2597 | -0.0793 |
| Brier | 0.0695 | 0.0716 | +0.0021 |
