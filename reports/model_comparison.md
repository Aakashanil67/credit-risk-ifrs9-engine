# Baseline vs LightGBM

LightGBM best params from 5-fold CV: `{'learning_rate': 0.05, 'num_leaves': 31}`, stopped at 264 trees via early stopping against validation AUC.

| metric | logistic baseline | LightGBM | delta |
|---|---|---|---|
| AUC | 0.7326 | 0.7565 | +0.0239 |
| Gini | 0.4652 | 0.5129 | +0.0478 |
| KS | 0.3466 | 0.3794 | +0.0328 |
