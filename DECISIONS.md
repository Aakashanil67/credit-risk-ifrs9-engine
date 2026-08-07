# Engineering decisions log

Why the code looks the way it does. Written as I went, so the reasoning is what I actually had at
the time rather than a tidy reconstruction. Most of this is the stuff that would otherwise live
only in my head: which approach I rejected, which library fought me, which number I tuned and why.

## Module map

| Module | Responsibility |
|---|---|
| `src/data_loader.py` | Loads `data/application_train.csv`; reports shape, target balance, missing-value counts. |
| `src/eda.py` | EDA charts plus `reports/eda_summary.md`. |
| `src/preprocessing.py` | Stratified 60/20/20 split, performed *before* imputation. sklearn `Pipeline` fits median/mode imputers on the training fold only. |
| `src/baseline.py` | statsmodels logistic regression, odds-ratio interpretation. |
| `src/features.py` | `build_lgbm_features()`. Split out of `train_lgbm.py` so the serving path stays clear of training-only imports. |
| `src/train_lgbm.py` | LightGBM, 5-fold CV, light tuning, AUC/Gini/KS against the baseline. |
| `src/scorecard.py` | optbinning WoE/IV scorecard and PDO points table. |
| `src/explain.py` | SHAP global and per-applicant, `reason_codes()`, calibration curve, Brier score. |
| `src/ecl.py` | IFRS 9 staged ECL. `PD × LGD × EAD`, staged 1/2/3 off a SICR proxy. |
| `api/main.py` | FastAPI `/predict` and `/health`. Live at `credit-risk-api-92it.onrender.com`. |
| `app/dashboard.py` | Streamlit UI. Calls the API, falls back to local scoring. Live at `credit-risk-ifrs9-engine.streamlit.app`. |

## Modelling decisions

**Split before impute, always.** The imputer is fitted on the training fold and only then applied
to validation and test. Doing it the other way round leaks the holdout's median back into training
and inflates every metric downstream. This is the single decision in the pipeline I would defend
hardest, because it is invisible in the results if you get it wrong: the numbers just come out
better and you never find out why.

**Accuracy is not reported anywhere.** 8.1% of applicants defaulted, so a model that answers
"repaid" to everything scores 91.9% while catching not a single default the lender actually loses
money on. AUC, Gini and KS run throughout instead, and the model card prints the base rate beside
every one of them so nobody reads 0.7565 as though it were an accuracy figure.

**LGD flat at 45%, EAD as the original credit amount.** Neither is fitted. This dataset carries no
recovery data at all, so anything I claimed there would be invented. Both sit in `src/config.py` as
named constants so they read as the assumptions they are.

**SICR staging compares two models, not one loan over time.** IFRS 9 wants a loan's PD today
against its own PD at origination. Kaggle's data is one static snapshot with no repeat observations,
so I proxy origination risk with the logistic baseline and current risk with the tuned LightGBM.
It demonstrates the staging mechanics correctly and it is not real risk migration. Documented in
`reports/ifrs9_summary.md` rather than buried.

**Reason codes say "missing", never "low", for a NaN feature.** A first-time applicant has no
bureau file. Telling them their score is low would be a specific false claim about a real person's
credit history, which is worse than a vague true one.

**Gender stays in as a direct feature, flagged.** It produces a 15.7-point approval-rate gap
(72.6% F against 56.9% M at an 8% cutoff). I left it in and wrote it up rather than dropping it
quietly, because dropping it does not fix the disparity: income, occupation and region reconstruct
most of the same signal. The honest position is that this needs a disparate-impact review, not a
one-line deletion.

## Things that fought me

**scipy has to stay at 1.13.1.** statsmodels 0.14.4 reaches for `_lazywhere`, which newer scipy
releases moved out of that import path, so letting scipy float to latest kills the baseline the
moment it loads. Pinned, with this note attached, because the symptom is a bare `ImportError`
naming a private function nobody recognises and pointing nowhere near statsmodels. It cost me
twenty minutes. Twice.

**optbinning segfaults, and the trigger is import order.** `BinningProcess` and `Scorecard` crashed
the interpreter outright on my machine. Not an exception, exit code 139, no traceback. I reproduced
it on synthetic data down to two numeric columns before I believed it. Two separate things were
going on. The solver was the first: optbinning defaults to OR-Tools CP-SAT, and the installed
ortools build has a broken `LinearExpr.__radd__` overload that crashes rather than raising. Driving
`OptimalBinning(solver="mip")` per feature and doing the PDO points maths by hand avoids the
orchestrator entirely.

The second one cost me most of the debugging time. Even with the MIP solver, importing numpy and
pandas *before* optbinning still segfaulted; importing optbinning first was reliably fine. ortools
ships its own native DLLs for abseil, protobuf and highs, and something in that set gets corrupted
by the other import order. `src/scorecard.py` therefore imports optbinning on line 28, above numpy
and pandas, with a `noqa` on the ruff isort rule. It looks wrong. It is load-bearing.

**optbinning's "Totals" row is an index label.** In `BinningTable.build()` output the summary row
carries `Bin == ''`, and the string "Totals" sits in the index instead, so the obvious filter on
the column quietly leaves the totals row inside every aggregate you then compute from it. Nothing
errors. Your IVs are just wrong. Filter on the index.

**Streamlit Community Cloud crash-looped on a Python version mismatch.** The build sat at "Resolved
68 packages" for over twenty minutes, re-provisioning the machine every 30 to 45 seconds, with
nothing in the log saying "failed". The platform defaulted to Python 3.14.6 and `numpy==1.26.4` has
no wheel for 3.14, so it was silently retrying a source build. Fixed in the app's own settings, not
in this repo: Settings → General → Python version, pinned to 3.12 to match the venv, CI and both
Docker images. If a Streamlit Cloud build ever seems to hang past a couple of minutes, check that
dropdown first.

**Render deployed a container with no model in it.** First deploy built fine and died at startup on
`FileNotFoundError: Missing model artifacts`. The `models/*.joblib` files were committed, but
neither Dockerfile actually copied them in. Locally this never showed up because `docker-compose.yml`
volume-mounts `./models` over that path, which papers over the gap. Render builds the Dockerfile
directly with no host filesystem to mount from. Added `COPY models/ models/` to both, then verified
with a standalone `docker build` and `docker run` with no compose and no volume, which is the only
way to reproduce Render's actual conditions.

**Streamlit's resolver failed on a dependency the dashboard never uses.** `uv` reported no solution:
optbinning 0.19.0 wants `ortools>=9.4`, which wants `numpy>=2.0.2`, against my `numpy==1.26.4` pin.
optbinning is only used by `src/scorecard.py`, which the dashboard never touches. It was being
dragged in transitively through `api/scoring.py` → `src/explain.py` → `src/train_lgbm.py` →
`src/baseline.py` → statsmodels, because Python executes the whole import chain at module load
regardless of which functions get called. I split `build_lgbm_features()` into `src/features.py` so
the serving path stops importing the training path. Checked it properly afterwards rather than
assuming: importing `api.scoring` and grepping `sys.modules` for statsmodels, mlflow, optbinning,
ortools and cvxpy now returns nothing. `app/requirements.txt` is a second, independent guard, since
Streamlit Cloud prefers a requirements file in the app's own directory over the repo root.

## Smaller calls

**Pinned versions, not ranges.** `pip install -r requirements.txt` reproduces the environment this
was actually built and tested against. Ranges would have let scipy drift and rebroken statsmodels.

**`pythonpath = ["."]` in `pyproject.toml`** so pytest imports `src.*` as a package without an
editable install.

**`tests/test_api.py` skips itself when `models/` is empty.** Those tests need real trained
artifacts, which do not exist on a fresh CI checkout because there is no Kaggle data there. My first
two pushes after adding the file had CI failing on a `FileNotFoundError` that had nothing to do with
the change being pushed. A module-level `pytest.mark.skipif` fixes it: real locally, skipped in CI.
I checked both states by renaming `models/` away and rerunning, rather than assuming the guard
worked.

**Model artifacts are committed.** About 944KB across three `.joblib` files, with a `!models/*.joblib`
exception in `.gitignore`. Render and Streamlit Cloud both deploy from a clean checkout and neither
can run training first. These are learned parameters, not Kaggle's data, so committing them does not
run into the redistribution terms that committing `data/` would.

**xgboost dropped.** It was in the original dependency list and I never used it. LightGBM handles
the categorical columns natively, which is most of what I wanted from a second GBM.

## Open items

- The deployed dashboard scores locally rather than calling the live API. No `API_URL` secret is
  set on Streamlit Cloud, so it falls through to the same path it uses when the API is down. Both
  routes load identical artifacts and return identical predictions, so this is a code-path
  difference, not a correctness one. Setting the secret would close it.
- Root `requirements.txt` still carries statsmodels, mlflow and optbinning into both Docker images.
  Harmless there: those build on python:3.12-slim and never hit the resolver conflict, and CI builds
  both images on every push. Only the Streamlit Cloud path needed the leaner list.
- Render's free tier idles out, so the first request after a quiet spell waits on a cold start.
- No drift monitoring and no retraining schedule. The model is a snapshot and a live deployment
  would need both.

## Environment

- Python 3.12.10 in `.venv`. `\.venv\Scripts\activate` on PowerShell.
- Dataset at `data/application_train.csv`, gitignored, never committed.
- Remote: `github.com/Aakashanil67/credit-risk-ifrs9-engine`.
- Editor and tool state is ignored globally via `~/.gitignore_global`, not in this repo.
