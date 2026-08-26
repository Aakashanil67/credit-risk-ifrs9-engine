# Engineering decisions log

This file records decisions that change how the project should be read or extended. It is not a
release log; Git history covers individual changes.

## Current architecture

`src/train_lgbm.py` trains either an offline full profile or the 18-field application profile.
`models/application/` stores the model, category dtypes, fit-fold medians and metadata together.
The FastAPI service loads that bundle at startup. The Streamlit dashboard sends the same request
contract to the service, then uses the same bundle only when the service cannot be reached.

An unseen category is rejected at the API boundary rather than translated into a new ordinal code.
It may signal a source-system change or a value that did not exist when the model was fitted, and
converting it would produce a plausible score from data outside the model's contract, with no error
visible to the caller or the person reviewing its response.

`src/ecl_core.py` contains stage assignment and discounted scenario calculations. `src/ecl_demo.py`
sets the illustrative scenario weights and loss assumptions. The dashboard deliberately shows the
simpler 12-month calculation instead of presenting the demonstration as a production provision.

## Decisions that affect results

**The test fold stays untouched.** The data split is stratified 60/20/20 with seed 42. Imputation,
cross-validation and early stopping use only the fit and validation folds. The application
model's final AUC is 0.678678 and its Brier score is 0.071635 on the 61,503-row test fold. Those
numbers are less flattering than the older full-information result, but they answer the deployed
question.
That distinction matters.

That sequence is intentionally inconvenient: the final fold may contradict a more flattering
validation result, but preserving the separation is the only way to judge the reported estimate.

**The served contract excludes gender and bureau variables.** New applicants may not have a bureau
file, and a public form should not collect a protected attribute merely to score a demonstration.
`CODE_GENDER` remains in the raw data for offline group diagnostics. The fairness audit found an
approval-rate difference under the illustrative threshold, so removing the direct feature did not
end the review.

**The decision cutoff is an economics example.** A 12% performing margin, 2% operating cost, 2%
capital cost and 45% LGD yield a PD threshold of 0.140351. The formula lives in
`src/decision_policy.py`. It makes the dashboard reproducible, but it has no authority as a credit
policy.

**The model bundle is committed; the dataset is not.** Hosting providers build a fresh checkout and
cannot train the model first. The small fitted artefacts are therefore versioned in Git and copied
into both images. `application_train.csv` stays ignored because it is competition data.

## Things that failed and what changed

**A Docker mount hid a deployment defect.** Local Compose mounted `models/` over `/app/models`, so
the first Render build passed locally and failed in the platform with missing artefacts. Both
Dockerfiles now copy the model bundle into the image. A standalone container test confirms that the
API loads the bundle without a host mount.

**The old serving path reached the model-building stack.** `api.scoring` reached `src.explain`, which once
loaded MLflow and scorecard dependencies as a side effect. The serving path now has
its own `src/reason_codes.py`; `requirements/api.txt` installs only the pinned API runtime. The
full root requirements file remains for development tooling and the scorecard challenger.

**The scorecard library is platform-sensitive.** On this Windows environment, the `optbinning`
solver path can terminate the interpreter. The challenger runs successfully inside a
Linux container with `OptimalBinning(solver="mip")`. It is kept as an offline experiment and is not
part of the API image.

## Remaining gaps

- Streamlit must receive `API_URL=https://credit-risk-api-92it.onrender.com` in its deployment
  secrets before the public dashboard calls the hosted API by default.
- The project has no out-of-time validation, local outcome data, drift monitor or scheduled
  retraining process.
- The ECL scenarios demonstrate mechanics only. They lack observed recovery data, contractual cash
  flows and an account-level PD history.
