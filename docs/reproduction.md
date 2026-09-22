# Reproduction guide

This guide separates two jobs that are easy to confuse: running the published model bundle and rebuilding the research evidence from the raw Home Credit data.

## 1. Run the published bundle

The committed `models/public_demo/` bundle is enough to run the API and dashboard. Docker Desktop is the shortest path:

```powershell
git clone https://github.com/Aakashanil67/credit-risk-ifrs9-engine.git
cd credit-risk-ifrs9-engine
docker compose up --build
```

Use the dashboard at `http://localhost:8501` and the OpenAPI page at `http://localhost:8000/docs`. The dashboard calls the API over the Compose network. Stop both containers with:

```powershell
docker compose down
```

## 2. Prepare a research environment

Use Python 3.12. Model deserialisation depends on the pinned NumPy, pandas, scikit-learn and LightGBM versions, so a global or partially upgraded environment is not supported.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
python -m pip check
```

Download `application_train.csv` from the [Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk) and place it at `data/application_train.csv`. The competition files are excluded from Git and are not redistributed here.

## 3. Verify the checkout before retraining

```powershell
pytest -q
ruff check .
ruff format --check .
pre-commit run --all-files
```

These commands cover unit, API, dashboard, artefact-contract and report-generation behaviour. They do not prove model suitability for a new lending portfolio.

## 4. Rebuild the public-demo model

The training flow uses a deterministic stratified 60/20/20 split with seed 42. Cross-validation is confined to the training fold; the validation fold supplies early stopping; the test fold is opened for final evaluation only.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m src.train_lgbm --profile public_demo
& $py -m src.explain --profile public_demo
& $py -m src.public_demo_audit
```

Retraining overwrites the versioned `models/public_demo/` bundle. Do it on a branch and inspect the metric, schema and artefact changes before replacing a published release.

The main outputs are:

- `models/public_demo/model.joblib`, category dtypes and metadata;
- `reports/model_comparison.md` and explainability figures;
- `reports/public_demo_audit.json`, `fairness_audit.md` and `threshold_analysis.md`.

## 5. Rebuild the lifecycle evidence

```powershell
& $py -m src.challenger_validation
& $py -m src.monitoring_demo
& $py -m src.validation_report
```

Challenger selection uses out-of-fold predictions from development data only. The monitoring demonstration then divides the frozen test data into a 51,503-row reference window and a disjoint 10,000-row replay window. Those runs are deterministic simulations, not a live monitoring feed.

Generated outputs include `reports/challenger_validation.*`, `models/public_demo/monitoring_reference.json`, `reports/monitoring_demo.*` and `reports/validation_report.md`.

## 6. Rebuild the IFRS 9 mechanics example

```powershell
& $py -m src.ecl
```

This writes `reports/ifrs9_summary.md`. The inputs are stated assumptions rather than estimates from the Home Credit model. Check the month-level schedule and scenario reconciliation before changing any staging, PD, LGD, EAD or discount-rate assumption.

## 7. Run the services without Docker

Start these in separate terminals after activating the environment:

```powershell
uvicorn api.main:app --reload
```

```powershell
$env:API_URL = "http://localhost:8000"
streamlit run app/dashboard.py
```

An aggregate load check is available once the API is running:

```powershell
python scripts/load_test.py --url http://localhost:8000 --requests 15 --concurrency 3
```

The command exits non-zero on transport errors, unexpected HTTP responses, incomplete success counts or server errors. Add `--allow-rate-limit-test` only when deliberately testing the 20-requests-per-IP-per-minute limit.
