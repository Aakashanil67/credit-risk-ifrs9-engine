# Credit risk and IFRS 9 engine

This project scores a consumer-loan application with a probability-of-default model, returns SHAP
reason codes, and demonstrates the mechanics behind an expected-credit-loss calculation.

The [dashboard](https://credit-risk-ifrs9-engine.streamlit.app) is a public demonstration and the
[API documentation](https://credit-risk-api-92it.onrender.com/docs) exposes the same prediction
contract. Render's free service can take a short time to wake after inactivity.

## Why this is not an accuracy problem

The Home Credit training data contains 307,511 applications and an 8.1% default rate. A classifier
that labels every applicant as non-default would therefore achieve 91.9% accuracy while identifying
no defaults. This project reports AUC for discrimination and Brier score for calibration. Gini and
KS give credit-risk readers two familiar views of the same separation problem. Ranking and
calibration matter because the score is later used in an illustrative loss calculation.

## Served model

The deployed model uses 18 fields available at application time: loan details, income and employment
information, household details, selected categorical descriptors, regional population density and car
age. It does not accept gender or external credit-bureau scores. Those omissions matter. A
larger offline model can appear stronger because it sees information that a new application may not
have, so it would give a misleadingly optimistic demonstration of an origination workflow.

The application-profile model was selected with five-fold cross-validation on the training fold,
early-stopped on a separate validation fold, then evaluated once on the untouched 61,503-row test
fold.

| metric | untouched test result |
|---|---:|
| AUC | 0.678678 |
| Gini | 0.357355 |
| KS | 0.259713 |
| Brier score | 0.071635 |

`reports/model_comparison.md` records an older logistic benchmark with bureau fields. It is useful
as an interpretable challenger, but it is not an apples-to-apples comparison with the deployed
application model. The model card discusses that limitation directly.

## Run locally

Create a Python 3.12 environment and install the full development and training set:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download `application_train.csv` from the [Home Credit Default Risk Kaggle competition](https://www.kaggle.com/competitions/home-credit-default-risk)
and place it in `data/`. The dataset remains ignored by Git.

```powershell
python -m src.train_lgbm --profile application
python -m src.explain
python -m src.ecl

uvicorn api.main:app --reload
streamlit run app/dashboard.py
```

The trained application bundle is committed so that the API and dashboard can run from a clean
checkout. Retraining requires the Kaggle data; serving does not.

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"age_years":35,"years_employed":5,"income_total":180000,"credit_amount":450000,"annuity":22500,"owns_car":true,"owns_realty":true,"num_children":1,"family_members":3,"education":"Higher education"}'
```

To run the two services together:

```powershell
docker compose up --build
```

The API is available at `http://localhost:8000/docs` and the dashboard at
`http://localhost:8501`. Run `pytest -v` for tests and `ruff check . && ruff format --check .`
for linting and formatting checks.

## Design choices

- **Application contract before model complexity.** `src/model_profiles.py` fixes the serving
  fields and field order. `api/scoring.py` rejects unknown fields and categories outside the
  persisted training contract rather than silently recoding them.
- **Hold out the test fold.** Imputation, cross-validation and early stopping do not inspect the
  test data. The saved metadata records the exact feature list, package versions, data hash and
  test metrics used for the deployed artefact.
- **Decision threshold from stated assumptions.** Approval uses an illustrative expected-value
  rule: 12% performing margin, 2% operating cost, 2% capital cost and 45% LGD imply a PD cutoff
  of 0.140351. It is a transparent demonstration, not a lending policy.
- **Two ECL layers, both labelled.** The dashboard displays a simple 12-month `PD x LGD x EAD`
  estimate. `src/ecl_core.py` and `src/ecl_demo.py` demonstrate stage assignment, discounted cash
  flows and probability-weighted upside/base/downside scenarios. Neither substitutes for a
  production IFRS 9 impairment engine.
- **Protected attributes retained only for audit.** Gender is excluded from model features and the
  public request schema. `reports/fairness_audit.md` measures group calibration and approval rates
  offline, then states what the check cannot establish.

## Limits that matter

The data is not South African and has no realised recoveries, contractual repayment schedules or
repeat observations for the same loan. LGD, EAD and SICR are therefore
assumptions. The fairness audit covers gender and age bands on one historical split; it is not a
legal review or a production monitoring programme. There is no live drift-monitoring or retraining
process. The project should be read as a documented modelling exercise, not a system that can make
real lending decisions.

For the details behind those choices, start with [the model card](reports/model_card.md),
[the fairness audit](reports/fairness_audit.md), [the threshold analysis](reports/threshold_analysis.md)
and [the engineering decisions log](DECISIONS.md).
