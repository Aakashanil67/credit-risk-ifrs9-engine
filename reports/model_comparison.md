# Public-demo logistic baseline vs LightGBM

Both models use the same 15-field public-demo contract and the same untouched test fold. LightGBM parameters come from 5-fold CV on the training fold; early stopping on the validation fold selected 193 trees for `{'learning_rate': 0.05, 'num_leaves': 31}`. The final LightGBM model and logistic baseline were then each fitted on the combined train and validation folds before this test evaluation.

| metric | logistic baseline | LightGBM | delta |
|---|---|---|---|
| AUC | 0.6563 | 0.6774 | +0.0210 |
| Gini | 0.3127 | 0.3547 | +0.0420 |
| KS | 0.2319 | 0.2611 | +0.0291 |
| Brier | 0.0723 | 0.0716 | -0.0007 |
| PR_AUC | 0.1444 | 0.1612 | +0.0168 |
| LogLoss | 0.2691 | 0.2652 | -0.0039 |
