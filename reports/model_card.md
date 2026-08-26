# Model card: application-profile PD model

## Intended use

This is an educational probability-of-default model for an unsecured consumer-loan application.
It is exposed through a FastAPI endpoint and a Streamlit demonstration so that the model contract,
decision rule and explanation path can be inspected together. It is not approved for credit
decisions, product prices, collections, accounting provisions or automated adverse action.

## Data and split

The model uses Kaggle's Home Credit Default Risk `application_train.csv`: 307,511 historical loan
applications, 122 raw columns and an 8.1% default rate. `TARGET=1` records a late-payment outcome
defined by the competition data. The dataset is not South African.
That matters.

The data was split 60/20/20 with stratification and seed 42. All model selection happens on the
184,506-row training fold and the 61,502-row validation fold. The 61,503-row test fold is reserved
for the final estimate. The application bundle records its dataset hash, package versions, category
levels, feature names and test metrics in `models/application/metadata.json`.

## Inputs and output

The model receives 18 application-time fields. They include requested credit, annuity, declared
income, employment tenure, selected household fields, education, income type, family status,
occupation, employer type, regional population density and car age. Gender and external
credit-bureau fields are deliberately absent from the served contract.

The output is a PD estimate, three SHAP-based reason codes, an illustrative expected-value decision
and a simple 12-month loss estimate. A response is not a credit decision about a real person. The
reason codes explain model contributions; they do not establish causation.

## Performance

LightGBM used a five-fold training-fold sweep over three small parameter combinations. The selected
settings were `learning_rate=0.05` and `num_leaves=31`; early stopping selected 166 trees on the
validation fold. Performance on the untouched test fold was:

| metric | result |
|---|---:|
| AUC | 0.678678 |
| Gini | 0.357355 |
| KS | 0.259713 |
| Brier score | 0.071635 |

These numbers describe ranking and probability error on this one historical dataset. They are not
a promise of performance elsewhere. `reports/model_comparison.md` also shows a logistic benchmark
with a different, bureau-rich input set. Its stronger reported result should not be treated as a
head-to-head win or loss against this deployment profile.

One test fold is not stability evidence.

Because the test set comes from the same competition dataset and period as the training data, it
cannot show whether a changed applicant mix, product terms, data-capture process, macroeconomy or
recovery practice would alter calibration or approval rates; that needs out-of-time local monitoring.

## Decision and ECL assumptions

The public API approves an application below a PD threshold of 0.140351. The figure follows from
the project assumptions: 12% performing margin, 2% operating cost, 2% capital cost and 45% LGD.
Changing product pricing, capital treatment or LGD changes the cutoff. It is a teaching rule, not a
policy threshold.

The API's displayed loss estimate is `PD x 45% x requested credit`. The separate ECL demonstration
adds 12-month versus lifetime horizons, discounted cash flows and weighted macroeconomic scenarios.
Neither layer has observed recovery data, amortisation schedules or an account-level PD history, so
neither can be used for accounting provision.

## Fairness and monitoring

Gender is available only to the offline fairness audit. On the test fold, approval was 91.81% for
the F group and 84.02% for the M group under the illustrative threshold. Removing gender from the
model does not prove that proxy effects have disappeared. `reports/fairness_audit.md` reports the
group sizes, default rates, calibration error, true-positive rate and false-positive rate behind
that statement.

A real deployment would require local outcomes, governance review, independent validation,
adverse-action controls, stability monitoring, outcome feedback and a retraining-governance process.
Those controls are outside this repository.
