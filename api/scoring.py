"""Shared scoring logic between the FastAPI service and the Streamlit dashboard's offline fallback.

Kept separate from api/main.py so the dashboard can score an applicant directly against the saved
model artifacts (no HTTP round trip) using the exact same code path the live API uses, rather than
a second, drifting reimplementation of the same business logic.
"""

import joblib
import numpy as np
import pandas as pd
import shap

from api.schemas import ApplicantRequest, PredictResponse
from src.config import (
    CAT_DTYPES_PATH,
    DECISION_THRESHOLD,
    DEFAULT_LGD,
    LGBM_MODEL_PATH,
    TRAIN_MEDIANS_PATH,
)
from src.explain import reason_codes

REQUIRED_ARTIFACTS = [LGBM_MODEL_PATH, TRAIN_MEDIANS_PATH, CAT_DTYPES_PATH]


def load_artifacts() -> dict:
    missing = [p for p in REQUIRED_ARTIFACTS if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {[str(p) for p in missing]}. Run `python -m src.train_lgbm` first."
        )
    model = joblib.load(LGBM_MODEL_PATH)
    return {
        "model": model,
        "train_medians": joblib.load(TRAIN_MEDIANS_PATH),
        "cat_dtypes": joblib.load(CAT_DTYPES_PATH),
        "feature_names": model.feature_name_,
        "explainer": shap.TreeExplainer(model),
    }


def applicant_to_row(req: ApplicantRequest, feature_names: list[str], cat_dtypes: dict) -> pd.DataFrame:
    """Map the applicant form onto the model's raw Home Credit column names.

    Fields the form doesn't ask for (most notably EXT_SOURCE_1/2/3 — external credit-bureau
    scores a real system would fetch from a bureau API at application time, not ask the applicant
    for) are left as NaN. LightGBM was trained on genuinely incomplete data and handles this via
    its learned default split direction, the same as any other missing value.
    """
    raw = {
        "NAME_CONTRACT_TYPE": req.contract_type,
        "DAYS_BIRTH": -req.age_years * 365.25,
        "DAYS_EMPLOYED": -req.years_employed * 365.25 if req.years_employed is not None else np.nan,
        "AMT_INCOME_TOTAL": req.income_total,
        "AMT_CREDIT": req.credit_amount,
        "AMT_ANNUITY": req.annuity,
        "AMT_GOODS_PRICE": req.goods_price if req.goods_price is not None else req.credit_amount,
        "CODE_GENDER": req.gender,
        "FLAG_OWN_CAR": "Y" if req.owns_car else "N",
        "FLAG_OWN_REALTY": "Y" if req.owns_realty else "N",
        "CNT_CHILDREN": req.num_children,
        "CNT_FAM_MEMBERS": req.family_members,
        "NAME_EDUCATION_TYPE": req.education,
        "NAME_INCOME_TYPE": req.income_type,
        "NAME_FAMILY_STATUS": req.family_status,
        "OCCUPATION_TYPE": req.occupation,
        "ORGANIZATION_TYPE": req.organization_type,
        "REGION_POPULATION_RELATIVE": req.region_population_relative
        if req.region_population_relative is not None
        else np.nan,
        "OWN_CAR_AGE": req.own_car_age if req.own_car_age is not None else np.nan,
    }
    row = pd.DataFrame([raw]).reindex(columns=feature_names)
    for col, dtype in cat_dtypes.items():
        row[col] = row[col].astype(dtype)
    return row


def score_applicant(req: ApplicantRequest, artifacts: dict, lgd: float = DEFAULT_LGD) -> PredictResponse:
    row = applicant_to_row(req, artifacts["feature_names"], artifacts["cat_dtypes"])
    pd_estimate = float(artifacts["model"].predict_proba(row)[0, 1])
    decision = "decline" if pd_estimate >= DECISION_THRESHOLD else "approve"

    explanation = artifacts["explainer"](row)
    shap_row = pd.Series(explanation.values[0], index=row.columns)
    feature_row = row.iloc[0]
    codes = reason_codes(shap_row, feature_row, artifacts["train_medians"], top_n=3)

    # a brand-new application has no origination-time PD to compare against, so it's Stage 1 by
    # definition (see src/ecl.py for the staged, portfolio-level version used on existing loans)
    ecl = pd_estimate * lgd * req.credit_amount

    return PredictResponse(
        probability_of_default=round(pd_estimate, 4),
        decision=decision,
        decision_threshold=DECISION_THRESHOLD,
        reason_codes=codes,
        expected_credit_loss=round(ecl, 2),
        lgd_assumption=lgd,
    )
