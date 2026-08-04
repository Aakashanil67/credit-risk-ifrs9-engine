# Model card — credit-risk-ifrs9-engine

## Intended use

A probability-of-default (PD) model for unsecured consumer lending decisions, wrapped with a
WoE/IV scorecard, SHAP-based reason codes, and an IFRS 9 expected-credit-loss layer. Framed
throughout as if deployed at a South African lender, so the SICR staging, reason codes and
model-risk documentation follow SARB expectations for credit-scoring models; the underlying data
itself is not South African (see Limitations).

Not intended for: origination decisions made without a human in the loop, any lending secured
against property, or use outside the unsecured-consumer-credit context it was trained on.

## Data

Kaggle's Home Credit Default Risk competition, `application_train.csv`: 307,511 loan applications,
122 raw columns, one row per application. `TARGET` = 1 if the applicant had a payment more than a
threshold number of days late at some point (Home Credit's definition of default), 8.1% of the
sample. Split 60/20/20 (train/validation/test), stratified on `TARGET`, seed 42; the imputer and
every fitted transform are fit on the training fold only (see `src/preprocessing.py`).

## Performance

Validation set (61,502 applications), against the logistic regression baseline:

| metric | logistic baseline | LightGBM (this model) |
|---|---|---|
| AUC | 0.7326 | 0.7565 |
| Gini | 0.4652 | 0.5129 |
| KS | 0.3466 | 0.3794 |
| Brier score | — | 0.0678 |

LightGBM beats the baseline on every ranking metric. Brier score of 0.0678 sits below the
0.0743 a model would score by always predicting the base default rate (8.1%). The calibration
curve (`reports/figures/calibration_curve.png`) tracks the diagonal closely across the observed
PD range (0-30%), so predicted PDs can be read as real probabilities, which matters here
specifically because `ECL = PD x LGD x EAD` uses the raw PD value, not its rank.

The WoE scorecard's information value ranks external credit-bureau scores highest
(`EXT_SOURCE_3` IV 0.329, `EXT_SOURCE_2` IV 0.322, both "strong" by the conventional 0.3-0.5
band), consistent with SHAP's global importance ranking the same two features top of the
LightGBM model. See `reports/scorecard.md` and `reports/eda_summary.md`.

## Fairness considerations

Checked one thing directly rather than asserting the model is fair in general: predicted PD and
approval rate by `CODE_GENDER`, on the validation set, at an illustrative 8% PD approval cutoff.

| gender | n | mean predicted PD | actual default rate | approval rate at 8% cutoff |
|---|---|---|---|---|
| F | 40,677 | 7.12% | 7.02% | 72.6% |
| M | 20,823 | 9.95% | 10.13% | 56.9% |

The model is well-calibrated *within* each group (predicted PD tracks actual default rate closely
for both), so this isn't a calibration bug: it's the model faithfully reproducing a real gap in
observed default rates by gender in this training population. That's exactly the pattern that
draws regulatory attention regardless of intent: a 15.7-point approval-rate gap by gender is the
kind of thing a South African National Credit Act compliance review, or an SARB model-risk
assessor, would flag for further work before this went anywhere near a live decision.

`CODE_GENDER` is used directly as a scorecard and LightGBM feature. What this card does *not*
claim: that removing it fixes the disparity (proxy features like income, occupation and region
could reconstruct most of the same signal), or that a full disparate-impact test across other
protected characteristics has been run. It hasn't. Flagging this as the most important open item
before any production use, not glossing over it.

## Limitations

- **Not South African data.** Home Credit operates in several countries but this Kaggle release
  isn't SA-specific; the SARB/NCA framing throughout this repo is a deliberate exercise in
  "how would this be documented for a SA lender," not a claim about the underlying population.
  A real deployment needs local data and local outcome definitions.
- **IFRS 9 SICR staging is a proxy, not a real time series.** `src/ecl.py` compares the LightGBM
  model's PD against the logistic baseline's PD for the same applicant as a stand-in for
  "PD at origination," because this dataset is a single snapshot with no repeat observations of
  the same loan. See `reports/ifrs9_summary.md` for the full caveat.
- **Static training data.** Sampled at a point in time; no drift monitoring, no champion/challenger
  process, no scheduled retraining defined. A live model needs all three.
- **Gender used as a direct model input**, flagged above under Fairness considerations. It's the
  single item most likely to block a real compliance sign-off as-is.
- **LGD and EAD are simplifying assumptions**, not measured recovery data: LGD is a flat 45%
  configurable constant, EAD is the original credit amount rather than an amortised outstanding
  balance. Both are clearly labelled as assumptions in `reports/ifrs9_summary.md`, not fitted from
  observed recoveries (this dataset doesn't contain any).

## SARB model-risk framing

Under SARB's model-risk expectations (aligned with the Basel Committee's SR 11-7-style guidance
most SA banks already follow), a model like this would sit through: independent validation of the
LightGBM model against the logistic challenger (done here, informally, as the baseline
comparison), documented limitations and assumptions (this card), ongoing performance monitoring
against a stability threshold (not implemented; no live scoring feed exists yet), and periodic
model risk committee review before any staging or threshold change. Treat this repo as the
validation-and-documentation layer of that process, not a substitute for the governance around it.
