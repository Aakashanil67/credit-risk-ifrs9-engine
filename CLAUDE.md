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
- 2026-08-04 — Skeleton in place. Nothing modeled yet.
- 2026-08-04 — Data loader, EDA (5 findings, 3 figures), leakage-safe 60/20/20 split + imputation, logistic baseline (AUC 0.7326), LightGBM (AUC 0.7565, Gini 0.5129, KS 0.3794, beats baseline on all three), MLflow tracking with 4 logged runs (`run_experiments.py`) — all done and pushed.
- 2026-08-04 — Credit scorecard (`src/scorecard.py`, WoE/IV per feature computed manually via `OptimalBinning(solver="mip")`, PDO points table, EXT_SOURCE_2/3 the strongest IV at ~0.32 each), SHAP explainability (`src/explain.py`, global beeswarm/bar + per-applicant waterfall + `reason_codes()`), IFRS 9 staged ECL (`src/ecl.py`, portfolio EAD R36.75B / ECL R1.33B / coverage 3.614%, monotonic coverage by stage 3.2%→10.5%→32.0%) — all done, tested, and pushed.
- 2026-08-04 — Calibration + model card (Brier 0.0678, real gender fairness check: F 72.6% vs M 56.9% approval rate at the 8% cutoff — flagged, not hidden), FastAPI `/predict` (`api/main.py` + `api/scoring.py` shared with the dashboard, clean `field: message` 422 errors), Streamlit dashboard (`app/dashboard.py`, live-API-with-fallback verified in a real browser both ways), Dockerfiles + compose (untested locally — no Docker Desktop — but CI's `docker-build` job builds both images clean on every push, so they're real-verified even without local Docker), hardening (38 tests, pre-commit with ruff, edge cases, CI-skip guard for `test_api.py` when `models/` isn't present), recruiter-grade README — all done, tested, and pushed. Full 20-day AUTOPILOT scope is complete except deployment (see README Status section).
- 2026-08-04 — Statsmodels 0.14.4 needs scipy pinned to 1.13.1 (newer scipy dropped `_lazywhere` from the import path statsmodels 0.14.4 uses) — pinned in `requirements.txt`, don't let scipy drift to latest.
- 2026-08-04 — `optbinning.BinningProcess`/`Scorecard` segfault unconditionally on this machine (reproduced on synthetic data down to 2 numeric columns) — root cause is `optbinning`'s default CP-SAT solver hitting a broken `LinearExpr.__radd__` overload in the installed `ortools` version, occasionally crashing the process instead of raising. Worked around by calling `OptimalBinning(..., solver="mip")` per feature directly (`src/scorecard.py`) instead of the higher-level orchestrator, and computing the PDO points table by hand. Separately — and this one cost real debugging time — the crash also depends on **import order**: `numpy`/`pandas` imported before `optbinning` reliably segfaults even with `solver="mip"`; `optbinning` imported first is reliably fine. `src/scorecard.py` imports `optbinning` before `numpy`/`pandas` for this reason — don't reorder it.
- 2026-08-04 — In `optbinning`'s `BinningTable.build()` output, the `"Totals"` summary row is the **index label**, not the value of the `Bin` column (`Bin` is `''` for that row) — filter on the index, not `row["Bin"] == "Totals"`.

## Decisions log

- 2026-08-04 — Pinned dependency versions in `requirements.txt` rather than loose ranges, so `pip install -r requirements.txt` reproduces the exact environment this was built and tested against.
- 2026-08-04 — `pyproject.toml` sets `pythonpath = ["."]` for pytest so tests import `src.*` as a package without an editable install.
- 2026-08-04 — `tests/test_api.py` needs real model artifacts (`models/*.joblib`) that don't exist on a fresh CI checkout with no Kaggle data — first two pushes after adding it had CI failing on a `FileNotFoundError` unrelated to the actual code change. Fixed with `pytestmark = pytest.mark.skipif(not LGBM_MODEL_PATH.exists(), ...)` at module level: runs for real locally where the model exists, skips cleanly in CI instead of erroring. Verified both states directly (renamed `models/` away and reran pytest) before trusting it.

## Known issues / deferred

- Docker Desktop is not installed on this machine, so `docker compose up` (full networked
  run, both containers talking to each other) has never been exercised locally. CI's
  `docker-build` job does build both images clean on every push (ubuntu-latest runners ship
  Docker), which caught and helped fix real issues (see decisions log) — so the Dockerfiles
  are genuinely verified, just not the compose networking. `models/` is mounted read-only at
  runtime rather than baked into the image (gitignored, and Kaggle's terms don't allow
  redistributing the data to bake it from anyway) — run `python -m src.train_lgbm` on the host
  first.
- `requirements.txt` drags training-only packages (statsmodels, mlflow, scipy, optbinning) into
  the serving images because `api/scoring.py` imports `src.explain` -> `src.train_lgbm` ->
  `src.baseline`, and Python executes that whole import chain at module load even though the API
  only calls a few functions from it. A serving-only import path would shrink the image; not done
  since it touches several modules for a size win, not a correctness one.
- Render.com / Streamlit Community Cloud deployment needs the user's own accounts on those services — documented as a manual step, not executed here.

## Environment notes

- Python: `C:\Users\aakas\...\Credit Risk and IFRS 9 engine\.venv` (3.12.10). Activate with `.venv\Scripts\activate` (PowerShell) before running anything manually — Claude Code invokes `.venv\Scripts\python.exe` / `.venv\Scripts\pip.exe` directly instead.
- `gh` CLI and `python` are installed via winget but not yet on PATH in a fresh shell — each command that needs them re-adds `Machine`+`User` PATH first (see any recent `PowerShell`/`Bash` call in history for the exact prefix).
- Dataset path: `data/application_train.csv` (gitignored).
- GitHub remote: `https://github.com/Aakashanil67/credit-risk-ifrs9-engine.git`, `gh auth setup-git` already wired the credential helper — plain `git push` works.
