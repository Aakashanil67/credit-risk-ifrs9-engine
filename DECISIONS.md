# Engineering decisions log

This file records choices that change how the project should be interpreted or extended. Git
history records individual code changes.

## Current architecture

`src.train_lgbm` trains the 15-field `public_demo` profile. Its LightGBM bundle lives in
`models/public_demo/` with the fitted category dtypes, metadata, and model. The
FastAPI service loads that bundle once at startup. The Streamlit dashboard sends the same request
contract to the API and can score locally from the same bundle if the API cannot be reached.

The API rejects unknown fields and unseen categorical values. This is intentional: coercing a new
source-system category into an arbitrary code would return a plausible-looking score from data
outside the fitted contract. Cross-field checks also reject impossible employment duration,
annuity greater than requested credit, and household counts inconsistent with children.

`src.challenger_validation`, `src.public_demo_audit`, and `src.monitoring_demo` run offline and
write aggregate evidence into `reports/` plus `models/public_demo/monitoring_reference.json`.
`src.validation_report` joins those artefacts without reading applicant rows. `api.observability`
builds allowlisted request metadata for the FastAPI middleware.

`src.ecl_core` owns stage assignment and discounted scenario calculations. `src.ecl_demo` owns the
three stated scenarios. `src.ecl` writes fixed mechanics examples instead of presenting a static
competition dataset as a real loan portfolio.

## Decisions that affect results

**The public form defines the deployed model.** Earlier iterations used fields that the dashboard
did not collect, leaving visitors to score missing data by default. The public profile removes
organisation type, regional density, and car age. It also adds goods price and fitted categorical
selectors so the deployed form and saved model agree.

**The test fold remains untouched until final evaluation.** The split is stratified 60/20/20 with
seed 42. Five-fold CV chooses parameters on the training fold. Validation selects the tree count.
The selected model is refit on all development data before one final test score. The v1.2 bundle's
test metrics are stored with the artifact, not copied by hand into the service.

**The model comparison is like for like.** The logistic baseline and LightGBM both use the same
15 inputs, combined development data, and untouched test fold. A bureau-rich baseline may be a
useful offline challenger, but it is not a fair claim about a public application-time service.

**Amounts have no named currency.** Home Credit does not identify the currency of its monetary
fields. The API, dashboard, reports, and reason codes therefore say "dataset monetary units" and
do not use a rand symbol.

**The approval label is illustrative.** A 12% performing margin, 2% operating cost, 2% capital
cost, and 45% LGD imply a PD threshold of 0.140351. The formula is visible in
`src.decision_policy`; it is not a credit policy.

**Gender is audit-only.** The served model and request schema exclude `CODE_GENDER`. The offline
fairness report uses it to surface group differences, not to declare the model fair or unfair.

**Challengers stay on development data.** `src.challenger_validation` produces five-fold
out-of-fold predictions across 246,008 train-plus-validation rows and never reads the 61,503-row
test fold for candidate selection. Calibrators are cross-fitted a second time, so a calibration
model predicts only rows outside its own fitting fold.

**No challenger is promoted automatically.** Five candidates were assessed against predeclared
discrimination and calibration gates with 1,000 paired stratified 20,000-row resamples. None
passed every gate, so `reports/challenger_validation.json` records a null nomination and the
v1.2.0 bundle remains the deployed artifact.

**Monitoring references are aggregate and thresholds are configurable.** The reference stores
feature-bin proportions, category proportions, missingness, score distribution, and approval rate
from 246,008 development rows; it stores no applicant rows or identifiers. PSI amber/red cutoffs
of 0.10 and 0.25 are illustrative controls, not universal lending-policy limits.

**HTTP request logs use an allowlist.** `api.observability.request_log_record` emits only event,
request ID, method, path, status, duration, service version, and model version. Payloads, IP
addresses, query strings, predictions, and explanation text are not logged.

## Failures that informed the design

**A Docker mount hid a deployment defect.** Local Compose mounted `models/` over `/app/models`,
so the first standalone Render image lacked model artifacts. Both Dockerfiles now copy versioned
bundles into the image; Compose's mount is only a local-development override.

**The first Streamlit dependency set was broader than its runtime path.** Training-only packages
pulled an incompatible solver stack into Streamlit Cloud. The dashboard has a lean
`app/requirements.txt`, and serving imports no longer transitively load MLflow or scorecard code.

**LightGBM 4.7.0 was rejected after a Windows reload failure.** A freshly serialised model faulted
when its feature names were read in a new Python process. The project pins 4.6.0, which addresses
the earlier advisory and completed the same fresh-process check. This is a compatibility decision,
not a claim that 4.7.0 is unsafe everywhere.

**The initial ECL loop repeated default probability without survival.** Over long horizons it could
exceed EAD before LGD. The current calculation multiplies each monthly hazard by the probability
of survival to that month. Stage 3 is explicitly a simplified first discounted cash shortfall.

## Remaining limits

- There is no out-of-time validation, local outcome data, scheduled production feed, or scheduled retraining.
- The data lacks recoveries, amortisation schedules, and observed risk migration; ECL remains a
  mechanics demonstration.
- The illustrative decision economics have no lender pricing or capital calibration.
- The fairness diagnostic is descriptive and cannot replace legal, policy, or governance review.
