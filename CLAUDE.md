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
- `src/features.py` — `build_lgbm_features()`, split out of `src/train_lgbm.py` so the serving path doesn't transitively import training-only packages (statsmodels, mlflow, optbinning). Both `src/train_lgbm.py` and `src/explain.py`/`src/ecl.py` import from here now.
- `api/main.py` — FastAPI `/predict` (PD, decision, reason codes, ECL) + `/health`. Live at `https://credit-risk-api-92it.onrender.com`.
- `app/dashboard.py` — Streamlit UI, calls the API with a local-model fallback. `app/requirements.txt` is a serving-only dependency subset for Streamlit Community Cloud (see State log). Live at `https://credit-risk-ifrs9-engine.streamlit.app`.

## State

- 2026-08-04 — Environment set up: Python 3.12.10 in `.venv`, all pinned deps installed (see `requirements.txt`). GitHub CLI installed and authenticated as `Aakashanil67`.
- 2026-08-04 — Dataset: `data/application_train.csv` downloaded from Kaggle (Home Credit Default Risk), 307,511 rows × 122 columns. Gitignored, never committed.
- 2026-08-04 — GitHub repo created: `github.com/Aakashanil67/credit-risk-ifrs9-engine`, public.
- 2026-08-04 — Skeleton in place. Nothing modeled yet.
- 2026-08-04 — Data loader, EDA (5 findings, 3 figures), leakage-safe 60/20/20 split + imputation, logistic baseline (AUC 0.7326), LightGBM (AUC 0.7565, Gini 0.5129, KS 0.3794, beats baseline on all three), MLflow tracking with 4 logged runs (`run_experiments.py`) — all done and pushed.
- 2026-08-04 — Credit scorecard (`src/scorecard.py`, WoE/IV per feature computed manually via `OptimalBinning(solver="mip")`, PDO points table, EXT_SOURCE_2/3 the strongest IV at ~0.32 each), SHAP explainability (`src/explain.py`, global beeswarm/bar + per-applicant waterfall + `reason_codes()`), IFRS 9 staged ECL (`src/ecl.py`, portfolio EAD R36.75B / ECL R1.33B / coverage 3.614%, monotonic coverage by stage 3.2%→10.5%→32.0%) — all done, tested, and pushed.
- 2026-08-04 — Calibration + model card (Brier 0.0678, real gender fairness check: F 72.6% vs M 56.9% approval rate at the 8% cutoff — flagged, not hidden), FastAPI `/predict` (`api/main.py` + `api/scoring.py` shared with the dashboard, clean `field: message` 422 errors), Streamlit dashboard (`app/dashboard.py`, live-API-with-fallback verified in a real browser both ways), Dockerfiles + compose (untested locally — no Docker Desktop — but CI's `docker-build` job builds both images clean on every push, so they're real-verified even without local Docker), hardening (38 tests, pre-commit with ruff, edge cases, CI-skip guard for `test_api.py` when `models/` isn't present), recruiter-grade README — all done, tested, and pushed. Full 20-day AUTOPILOT scope is complete except deployment (see README Status section).
- 2026-08-04 — Docker Desktop installed; `docker compose up --build` run for real for the first time: both containers built and started, `curl localhost:8000/predict` returned a real prediction, and the dashboard scored an applicant showing "Scored via live API (http://api:8000)" — confirms the compose service-name DNS resolution and the full container-to-container network path actually work, not just the individual image builds CI already checked. Committed `models/*.joblib` (~944KB total) to the repo (previously gitignored) since Render/Streamlit Cloud deploy from a fresh checkout with no way to run training first — these are learned parameters, not the Kaggle dataset, so this doesn't run into Kaggle's redistribution restriction. Render Web Service `credit-risk-api` configured (Docker, `api/Dockerfile`, Free tier, `PORT=8000`, GitHub App scoped to just this repo) but blocked on Render's payment-card verification requirement — flagged to the user, not bypassed.
- 2026-08-04 — **Both services deployed live.** User added a card on Render (their own action, not mine) and confirmed; Render's first deploy attempt then failed at startup with `FileNotFoundError: Missing model artifacts` — root cause: neither Dockerfile had `COPY models/ models/`, they relied entirely on `docker-compose.yml`'s volume mount, which doesn't exist on a platform that just builds the Dockerfile directly with no host filesystem. Fixed both Dockerfiles, verified locally with a standalone `docker build`+`docker run` (no compose, no volume — same conditions as Render), pushed (`d29b459`), Render auto-redeployed successfully. `curl https://credit-risk-api-92it.onrender.com/health` → `{"status":"ok","model_loaded":true}`, and a full `/predict` call returns the identical prediction (PD 0.0835, decline, ECL R16,910) seen in every other environment (local venv, local Docker, Docker Compose). Separately, Streamlit Community Cloud's build failed with `uv`'s resolver reporting no solution for `optbinning`'s `ortools>=9.4` (needs `numpy>=2.0.2`) against this repo's `numpy==1.26.4` pin — root cause was `api/scoring.py` -> `src.explain` -> (then) `src.train_lgbm` -> `src.baseline` transitively importing statsmodels/mlflow/optbinning at module load, even though the dashboard never calls into any of it. Fixed for real, not worked around: split `build_lgbm_features()` out into `src/features.py` (new file) so `src/explain.py`/`src/ecl.py` import from there instead of `src/train_lgbm.py`, breaking the transitive chain (`python -c "import api.scoring"` then shows zero statsmodels/mlflow/optbinning/ortools/cvxpy modules loaded). Also added `app/requirements.txt` — Streamlit Cloud reads a requirements.txt from the app's own directory in preference to repo root — a lean serving-only subset (no optbinning/statsmodels/mlflow/pytest/ruff). Rebuild succeeded (`uv` resolved 68 packages cleanly this time) and the app is live at `https://credit-risk-ifrs9-engine.streamlit.app`.
- 2026-08-04 — Statsmodels 0.14.4 needs scipy pinned to 1.13.1 (newer scipy dropped `_lazywhere` from the import path statsmodels 0.14.4 uses) — pinned in `requirements.txt`, don't let scipy drift to latest.
- 2026-08-04 — `optbinning.BinningProcess`/`Scorecard` segfault unconditionally on this machine (reproduced on synthetic data down to 2 numeric columns) — root cause is `optbinning`'s default CP-SAT solver hitting a broken `LinearExpr.__radd__` overload in the installed `ortools` version, occasionally crashing the process instead of raising. Worked around by calling `OptimalBinning(..., solver="mip")` per feature directly (`src/scorecard.py`) instead of the higher-level orchestrator, and computing the PDO points table by hand. Separately — and this one cost real debugging time — the crash also depends on **import order**: `numpy`/`pandas` imported before `optbinning` reliably segfaults even with `solver="mip"`; `optbinning` imported first is reliably fine. `src/scorecard.py` imports `optbinning` before `numpy`/`pandas` for this reason — don't reorder it.
- 2026-08-04 — In `optbinning`'s `BinningTable.build()` output, the `"Totals"` summary row is the **index label**, not the value of the `Bin` column (`Bin` is `''` for that row) — filter on the index, not `row["Bin"] == "Totals"`.

## Decisions log

- 2026-08-04 — Pinned dependency versions in `requirements.txt` rather than loose ranges, so `pip install -r requirements.txt` reproduces the exact environment this was built and tested against.
- 2026-08-04 — `pyproject.toml` sets `pythonpath = ["."]` for pytest so tests import `src.*` as a package without an editable install.
- 2026-08-04 — `tests/test_api.py` needs real model artifacts (`models/*.joblib`) that don't exist on a fresh CI checkout with no Kaggle data — first two pushes after adding it had CI failing on a `FileNotFoundError` unrelated to the actual code change. Fixed with `pytestmark = pytest.mark.skipif(not LGBM_MODEL_PATH.exists(), ...)` at module level: runs for real locally where the model exists, skips cleanly in CI instead of erroring. Verified both states directly (renamed `models/` away and reran pytest) before trusting it.

## Known issues / deferred

- Root `requirements.txt` still drags training-only packages (statsmodels, mlflow, scipy,
  optbinning) into the Docker images — fine there since those run on Python 3.12-slim and never
  hit the numpy/ortools resolver conflict, and CI validates both images build clean on every
  push. Only the Streamlit Cloud path needed trimming (`app/requirements.txt`), since that
  platform's `uv` resolver is what actually failed.
- No `API_URL` secret set on Streamlit Community Cloud, so the deployed dashboard always uses its
  local-model fallback rather than calling the live Render API — `app/dashboard.py`'s default
  (`http://localhost:8000`) doesn't resolve on Streamlit Cloud's infrastructure. Both paths score
  identically (same model artifacts), so this doesn't affect correctness, just which code path
  runs. Could be fixed by setting `API_URL=https://credit-risk-api-92it.onrender.com` as a
  Streamlit Cloud secret if demoing the live-API path specifically matters.
- Render's free tier spins down on inactivity — first request after idle will be slow (cold
  start) before the model loads.

## Environment notes

- Python: `C:\Users\aakas\...\Credit Risk and IFRS 9 engine\.venv` (3.12.10). Activate with `.venv\Scripts\activate` (PowerShell) before running anything manually — Claude Code invokes `.venv\Scripts\python.exe` / `.venv\Scripts\pip.exe` directly instead.
- `gh` CLI and `python` are installed via winget but not yet on PATH in a fresh shell — each command that needs them re-adds `Machine`+`User` PATH first (see any recent `PowerShell`/`Bash` call in history for the exact prefix).
- Dataset path: `data/application_train.csv` (gitignored).
- GitHub remote: `https://github.com/Aakashanil67/credit-risk-ifrs9-engine.git`, `gh auth setup-git` already wired the credential helper — plain `git push` works.
