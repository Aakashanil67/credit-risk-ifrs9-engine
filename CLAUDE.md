# CLAUDE.md — project memory

Running context for this repo. Updated after each build step, not just at the end of a session — read this first before doing anything else here.

## Architecture

- `src/data_loader.py` — loads `data/application_train.csv`, reports shape/target balance/missing values.
- `src/eda.py` — EDA charts + `reports/eda_summary.md`.
- `src/preprocessing.py` — stratified 60/20/20 split (before imputation), sklearn `Pipeline` for median/mode imputation fitted on train only.
- `src/baseline.py` — statsmodels logistic regression, odds-ratio interpretation.
- `src/train_lgbm.py` — LightGBM, 5-fold CV, light tuning, AUC/Gini/KS vs baseline.
- `src/scorecard.py` — optbinning WoE/IV scorecard.
- `src/explain.py` — SHAP global + per-applicant, `reason_codes()`, calibration curve/Brier score.
- `src/ecl.py` — IFRS 9 staged ECL (`PD × LGD × EAD`, Stage 1/2/3 via SICR = PD doubled since origination).
- `api/main.py` — FastAPI `/predict` (PD, decision, reason codes, ECL) + `/health`.
- `app/dashboard.py` — Streamlit UI, calls the API with a local-model fallback.

## State

- 2026-08-04 — Environment set up: Python 3.12.10 in `.venv`, all pinned deps installed (see `requirements.txt`). GitHub CLI installed and authenticated as `Aakashanil67`.
- 2026-08-04 — Dataset: `data/application_train.csv` downloaded from Kaggle (Home Credit Default Risk), 307,511 rows × 122 columns. Gitignored, never committed.
- 2026-08-04 — GitHub repo created: `github.com/Aakashanil67/credit-risk-ifrs9-engine`, public.
- 2026-08-04 — Skeleton in place (this commit). Nothing modeled yet.

## Decisions log

- 2026-08-04 — Pinned dependency versions in `requirements.txt` rather than loose ranges, so `pip install -r requirements.txt` reproduces the exact environment this was built and tested against.
- 2026-08-04 — `pyproject.toml` sets `pythonpath = ["."]` for pytest so tests import `src.*` as a package without an editable install.

## Known issues / deferred

- Docker Desktop is not installed on this machine. Dockerfiles will be written and explained but not built/run unless that changes.
- Render.com / Streamlit Community Cloud deployment needs the user's own accounts on those services — documented as a manual step, not executed here.

## Environment notes

- Python: `C:\Users\aakas\...\Credit Risk and IFRS 9 engine\.venv` (3.12.10). Activate with `.venv\Scripts\activate` (PowerShell) before running anything manually — Claude Code invokes `.venv\Scripts\python.exe` / `.venv\Scripts\pip.exe` directly instead.
- `gh` CLI and `python` are installed via winget but not yet on PATH in a fresh shell — each command that needs them re-adds `Machine`+`User` PATH first (see any recent `PowerShell`/`Bash` call in history for the exact prefix).
- Dataset path: `data/application_train.csv` (gitignored).
- GitHub remote: `https://github.com/Aakashanil67/credit-risk-ifrs9-engine.git`, `gh auth setup-git` already wired the credential helper — plain `git push` works.
