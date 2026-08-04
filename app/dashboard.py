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
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.schemas import ApplicantRequest  # noqa: E402
from api.scoring import applicant_to_row, load_artifacts, score_applicant  # noqa: E402
from src.config import DEFAULT_EAD_COL, DEFAULT_LGD  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_TIMEOUT_SECONDS = 3.0

st.set_page_config(page_title="Credit Risk & IFRS 9 Engine", layout="wide")


@st.cache_resource
def get_local_artifacts():
    return load_artifacts()


def call_api(payload: dict) -> dict | None:
    try:
        response = httpx.post(f"{API_URL}/predict", json=payload, timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        return None


def score_locally(req: ApplicantRequest, lgd: float) -> dict:
    artifacts = get_local_artifacts()
    result = score_applicant(req, artifacts, lgd=lgd)
    return result.model_dump()


st.title("Credit Risk & IFRS 9 Engine")
st.caption("Score a loan applicant: probability of default, decision, SHAP reason codes, IFRS 9 ECL.")

with st.sidebar:
    st.header("Provisioning assumptions")
    lgd = st.slider(
        "LGD — loss given default", min_value=0.0, max_value=1.0, value=DEFAULT_LGD, step=0.05,
        help="Share of exposure not recovered after a default. Applied on top of whatever PD the model returns.",
    )
    st.caption(f"EAD is the applicant's requested credit amount (`{DEFAULT_EAD_COL}`) — set it in the form.")
    st.divider()
    st.caption(f"API: `{API_URL}`")

with st.form("applicant_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Applicant")
        age_years = st.number_input("Age (years)", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["F", "M"])
        num_children = st.number_input("Number of children", min_value=0, max_value=20, value=0)
        family_members = st.number_input("Family members", min_value=1, max_value=20, value=1)
        family_status = st.selectbox(
            "Family status",
            ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"],
        )

    with col2:
        st.subheader("Employment & income")
        years_employed = st.number_input(
            "Years employed (leave 0 if not currently employed)", min_value=0.0, max_value=60.0, value=5.0
        )
        income_type = st.selectbox(
            "Income type", ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
        )
        income_total = st.number_input("Annual income (R)", min_value=1.0, value=180_000.0, step=10_000.0)
        education = st.selectbox(
            "Education",
            ["Secondary / secondary special", "Higher education", "Incomplete higher",
             "Lower secondary", "Academic degree"],
        )
        occupation = st.text_input("Occupation (optional)", value="")

    with col3:
        st.subheader("Loan")
        contract_type = st.selectbox("Loan type", ["Cash loans", "Revolving loans"])
        credit_amount = st.number_input("Credit amount (R)", min_value=1.0, value=450_000.0, step=10_000.0)
        annuity = st.number_input("Monthly annuity (R)", min_value=1.0, value=22_500.0, step=500.0)
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
        gender=gender,
        owns_car=owns_car,
        owns_realty=owns_realty,
        num_children=num_children,
        family_members=family_members,
        education=education,
        income_type=income_type,
        family_status=family_status,
        occupation=occupation or None,
    )

    api_result = call_api(req.model_dump())
    if api_result is not None:
        st.success(f"Scored via live API ({API_URL})")
        result = api_result
        # the API always uses the configured DEFAULT_LGD — recompute ECL locally if the sidebar
        # LGD differs, so the slider actually does something even when the API is reachable
        if abs(lgd - result["lgd_assumption"]) > 1e-9:
            result["expected_credit_loss"] = round(result["probability_of_default"] * lgd * credit_amount, 2)
            result["lgd_assumption"] = lgd
    else:
        st.info("API unreachable — scoring directly against the saved model artifacts instead.")
        result = score_locally(req, lgd)

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
            st.success(f"**APPROVE** — PD {pd_estimate:.1%} is below the {result['decision_threshold']:.0%} cutoff")
        else:
            st.error(f"**DECLINE** — PD {pd_estimate:.1%} is at or above the {result['decision_threshold']:.0%} cutoff")

        st.metric("Expected credit loss (12-month, Stage 1)", f"R{result['expected_credit_loss']:,.2f}")
        st.caption(f"ECL = PD x LGD ({result['lgd_assumption']:.0%}) x credit amount (R{credit_amount:,.0f})")

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
            st.caption("Local model artifacts unavailable — waterfall plot needs `python -m src.train_lgbm` run once.")
