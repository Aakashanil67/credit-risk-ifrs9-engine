# Scorecard challenger

`src/scorecard.py` is an offline WoE/IV challenger, not the model served by the API. It uses three
bureau fields that are intentionally excluded from the application profile, so its results should
not be used to explain a dashboard prediction or compared directly with the deployed model.
It is not deployed.

The challenger fits each bin with `OptimalBinning(solver="mip")`, transforms it to weight of
evidence, and then scales logistic-regression contributions to a 600-point, 50:1 good-to-bad score
with 20 points to double the odds. In a Linux container, its validation scores ranged from 471 to
667 with a mean of 566.4. The highest-information variables were `EXT_SOURCE_3` (IV 0.328917),
`EXT_SOURCE_2` (0.321753) and `EXT_SOURCE_1` (0.145918).

The earlier, full points table was retired when the application model stopped using gender and
bureau data. It would have implied that it described the public service. Run `python -m
src.scorecard` with the Kaggle data to regenerate an exploratory table for the challenger.
