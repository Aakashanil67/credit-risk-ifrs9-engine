"""Scoring shared by FastAPI and the dashboard's local fallback."""

import numpy as np
import pandas as pd
import shap

from api.schemas import ApplicantRequest, PredictResponse
from src.artifacts import load_artifact_bundle
from src.config import (
    CAPITAL_COST_RATE,
    DEFAULT_LGD,
    OPERATING_COST_RATE,
    PERFORMING_MARGIN_RATE,
    model_bundle_dir,
)
from src.decision_policy import DecisionPolicy
from src.reason_codes import reason_codes


class InvalidApplicantError(ValueError):
    """A supplied category is outside the fitted public-demo contract."""

    def __init__(self, field: str, detail: str) -> None:
        super().__init__(detail)
        self.field = field


CATEGORY_FIELD_NAMES = {
    "NAME_CONTRACT_TYPE": "contract_type",
    "NAME_EDUCATION_TYPE": "education",
    "NAME_INCOME_TYPE": "income_type",
    "NAME_FAMILY_STATUS": "family_status",
    "OCCUPATION_TYPE": "occupation",
}

NUMERIC_FEATURES = {
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
}


def load_artifacts() -> dict:
    bundle = load_artifact_bundle(model_bundle_dir("public_demo"))
    return {
        "model": bundle.model,
        "cat_dtypes": bundle.category_dtypes,
        "metadata": bundle.metadata,
        "feature_names": bundle.metadata["feature_names"],
        "explainer": shap.TreeExplainer(bundle.model),
    }


def applicant_to_row(
    req: ApplicantRequest, feature_names: list[str], cat_dtypes: dict
) -> pd.DataFrame:
    raw = {
        "NAME_CONTRACT_TYPE": req.contract_type,
        "DAYS_BIRTH": -req.age_years * 365.25,
        "DAYS_EMPLOYED": -req.years_employed * 365.25 if req.years_employed is not None else np.nan,
        "AMT_INCOME_TOTAL": req.income_total,
        "AMT_CREDIT": req.credit_amount,
        "AMT_ANNUITY": req.annuity,
        "AMT_GOODS_PRICE": req.goods_price if req.goods_price is not None else req.credit_amount,
        "FLAG_OWN_CAR": "Y" if req.owns_car else "N",
        "FLAG_OWN_REALTY": "Y" if req.owns_realty else "N",
        "CNT_CHILDREN": req.num_children,
        "CNT_FAM_MEMBERS": req.family_members,
        "NAME_EDUCATION_TYPE": req.education,
        "NAME_INCOME_TYPE": req.income_type,
        "NAME_FAMILY_STATUS": req.family_status,
        "OCCUPATION_TYPE": req.occupation,
    }
    row = pd.DataFrame([raw]).reindex(columns=feature_names)
    for column in NUMERIC_FEATURES.intersection(row.columns):
        row[column] = pd.to_numeric(row[column], errors="coerce")
    for column, dtype in cat_dtypes.items():
        value = row.at[0, column]
        if pd.notna(value) and value not in dtype.categories:
            field = CATEGORY_FIELD_NAMES.get(column, column)
            raise InvalidApplicantError(field, f"unsupported category {value!r}")
        row[column] = row[column].astype(dtype)
    return row


def score_applicant(
    req: ApplicantRequest, artifacts: dict, lgd: float = DEFAULT_LGD
) -> PredictResponse:
    row = applicant_to_row(req, artifacts["feature_names"], artifacts["cat_dtypes"])
    pd_estimate = float(artifacts["model"].predict_proba(row)[0, 1])
    policy = DecisionPolicy(
        margin_rate=PERFORMING_MARGIN_RATE,
        operating_cost_rate=OPERATING_COST_RATE,
        capital_cost_rate=CAPITAL_COST_RATE,
        lgd=lgd,
    )

    explanation = artifacts["explainer"](row)
    shap_row = pd.Series(explanation.values[0], index=row.columns)
    codes = reason_codes(shap_row, row.iloc[0], top_n=3)

    # This is an illustrative 12-month loss estimate, not a portfolio IFRS 9 calculation.
    ecl = pd_estimate * lgd * req.credit_amount
    metadata = artifacts["metadata"]
    return PredictResponse(
        probability_of_default=round(pd_estimate, 4),
        decision=policy.decision(pd_estimate),
        decision_threshold=policy.threshold,
        reason_codes=codes,
        expected_credit_loss=round(ecl, 2),
        lgd_assumption=lgd,
        expected_value=round(policy.expected_value(pd_estimate, req.credit_amount), 2),
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
        model_profile=metadata["profile"],
    )
