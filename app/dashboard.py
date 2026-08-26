"""Streamlit decision dashboard: fill in an applicant, get a scored decision with an explanation.

Calls the FastAPI service first (configurable via the API_URL env var). If the API is unreachable
— not running, still starting up, whatever — it falls back to scoring directly against the saved
model artifacts via api.scoring, so the dashboard still works standalone.
"""

import os
import sys
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.schemas import ApplicantRequest  # noqa: E402
from api.scoring import applicant_to_row, load_artifacts, score_applicant  # noqa: E402
from src.config import DEFAULT_EAD_COL  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_TIMEOUT_SECONDS = 10.0

st.set_page_config(page_title="Credit Risk & IFRS 9 Engine", layout="wide")


@st.cache_resource
def get_local_artifacts():
    return load_artifacts()


def call_api(payload: dict) -> dict | None:
    try:
        response = httpx.post(f"{API_URL}/predict", json=payload, timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except (httpx.ConnectError, httpx.TimeoutException):
        return None


def score_locally(req: ApplicantRequest) -> dict:
    artifacts = get_local_artifacts()
    result = score_applicant(req, artifacts)
    return result.model_dump()


def category_options(column: str, fallback: list[str]) -> list[str]:
    """Use fitted categories so the public form cannot send an unsupported label."""
    try:
        categories = get_local_artifacts()["cat_dtypes"][column].categories.tolist()
    except (FileNotFoundError, KeyError):
        return fallback
    return [str(category) for category in categories if pd.notna(category)]


st.title("Credit Risk & IFRS 9 Engine")
st.caption("Public-demo PD scoring, illustrative decisions, and explanation codes.")

with st.sidebar:
    st.header("Model contract")
    st.caption(
        f"The loss example uses the requested credit amount (`{DEFAULT_EAD_COL}`) as EAD and a fixed 45% LGD. "
        "Amounts are dataset monetary units; the source dataset does not identify a currency."
    )
    st.divider()
    st.caption(f"API: `{API_URL}`")

with st.form("applicant_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Applicant")
        age_years = st.number_input("Age (years)", min_value=18, max_value=100, value=35)
        num_children = st.number_input("Number of children", min_value=0, max_value=20, value=0)
        family_members = st.number_input("Family members", min_value=1, max_value=20, value=1)
        family_status = st.selectbox(
            "Family status",
            category_options(
                "NAME_FAMILY_STATUS",
                ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"],
            ),
        )

    with col2:
        st.subheader("Employment & income")
        years_employed = st.number_input(
            "Years employed (leave 0 if not currently employed)",
            min_value=0.0,
            max_value=60.0,
            value=5.0,
        )
        income_type = st.selectbox(
            "Income type",
            category_options(
                "NAME_INCOME_TYPE",
                ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
            ),
        )
        income_total = st.number_input(
            "Annual income (dataset monetary units)", min_value=1.0, value=180_000.0, step=10_000.0
        )
        education = st.selectbox(
            "Education",
            category_options(
                "NAME_EDUCATION_TYPE",
                [
                    "Secondary / secondary special",
                    "Higher education",
                    "Incomplete higher",
                    "Lower secondary",
                    "Academic degree",
                ],
            ),
        )
        occupation = st.selectbox(
            "Occupation",
            ["Not provided", *category_options("OCCUPATION_TYPE", ["Laborers", "Sales staff", "Managers"])],
        )

    with col3:
        st.subheader("Loan")
        contract_type = st.selectbox(
            "Loan type", category_options("NAME_CONTRACT_TYPE", ["Cash loans", "Revolving loans"])
        )
        credit_amount = st.number_input(
            "Credit amount (dataset monetary units)", min_value=1.0, value=450_000.0, step=10_000.0
        )
        goods_price = st.number_input(
            "Goods price, if applicable (dataset monetary units)",
            min_value=1.0,
            value=450_000.0,
            step=10_000.0,
        )
        annuity = st.number_input(
            "Monthly annuity (dataset monetary units)", min_value=1.0, value=22_500.0, step=500.0
        )
        owns_car = st.checkbox("Owns a car")
        owns_realty = st.checkbox("Owns property")

    submitted = st.form_submit_button("Score applicant")

if submitted:
    req = ApplicantRequest(
        contract_type=contract_type,
        age_years=age_years,
        years_employed=years_employed if years_employed > 0 else None,
        income_total=income_total,
        credit_amount=credit_amount,
        annuity=annuity,
        goods_price=goods_price,
        owns_car=owns_car,
        owns_realty=owns_realty,
        num_children=num_children,
        family_members=family_members,
        education=education,
        income_type=income_type,
        family_status=family_status,
        occupation=None if occupation == "Not provided" else occupation,
    )

    try:
        api_result = call_api(req.model_dump())
    except httpx.HTTPStatusError as exc:
        st.error(f"The API rejected this request: {exc.response.text}")
        st.stop()
    if api_result is not None:
        st.success(f"Scored via live API ({API_URL})")
        result = api_result
    else:
        st.info("API unreachable — scoring directly against the saved model artifacts instead.")
        result = score_locally(req)

    pd_estimate = result["probability_of_default"]
    decision = result["decision"]

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Decision")
        fig, ax = plt.subplots(figsize=(4, 3))
        color = "#c1121f" if decision == "decline" else "#2a6f97"
        ax.barh([0], [pd_estimate], color=color)
        ax.barh([0], [1], color="none", edgecolor="black", linewidth=0.5)
        ax.axvline(result["decision_threshold"], color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("probability of default")
        ax.set_title(f"PD = {pd_estimate:.1%}  (threshold {result['decision_threshold']:.0%})")
        st.pyplot(fig)
        plt.close(fig)

        if decision == "approve":
            st.success(
                f"**APPROVE** — PD {pd_estimate:.1%} is below the {result['decision_threshold']:.0%} cutoff"
            )
        else:
            st.error(
                f"**DECLINE** — PD {pd_estimate:.1%} is at or above the {result['decision_threshold']:.0%} cutoff"
            )

        st.metric(
            "Illustrative 12-month loss estimate",
            f"{result['expected_credit_loss']:,.2f} monetary units",
        )
        st.caption(
            f"ECL = PD x LGD ({result['lgd_assumption']:.0%}) x credit amount ({credit_amount:,.0f} monetary units)"
        )
        st.metric("Illustrative expected value", f"{result['expected_value']:,.2f} monetary units")
        st.caption(
            f"{result['model_name']} v{result['model_version']} · {result['model_profile']} profile"
        )

    with right:
        st.subheader("Why the model said this")
        for code in result["reason_codes"]:
            st.markdown(f"- {code}")

        st.subheader("SHAP waterfall")
        try:
            artifacts = get_local_artifacts()
            row = applicant_to_row(req, artifacts["feature_names"], artifacts["cat_dtypes"])
            explanation = artifacts["explainer"](row)
            fig = plt.figure()
            shap.plots.waterfall(explanation[0], show=False, max_display=8)
            st.pyplot(fig, bbox_inches="tight")
            plt.close(fig)
        except FileNotFoundError:
            st.caption(
                "Local model artifacts unavailable — waterfall plot needs `python -m src.train_lgbm` run once."
            )
