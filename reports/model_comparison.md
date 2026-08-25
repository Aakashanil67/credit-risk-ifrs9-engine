# Baseline vs LightGBM

LightGBM best params from 5-fold CV: `{'learning_rate': 0.05, 'num_leaves': 31}`, stopped at 166 trees via early stopping against validation AUC.

| metric | logistic baseline | LightGBM | delta |
|---|---|---|---|
| AUC | 0.7283 | 0.6787 | -0.0497 |
| Gini | 0.4567 | 0.3574 | -0.0993 |
| KS | 0.3390 | 0.2597 | -0.0793 |
| Brier | 0.0695 | 0.0716 | +0.0021 |
