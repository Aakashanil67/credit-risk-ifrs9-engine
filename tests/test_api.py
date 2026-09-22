import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, prediction_rate_limiter
from api.observability import SAFE_LOG_FIELDS
from api.schemas import ApplicantRequest
from api.scoring import applicant_to_row, load_artifacts
from src.config import model_bundle_dir

# These are integration tests against the real trained model, not unit tests — they need
# artifacts produced by `python -m src.train_lgbm`, which needs the Kaggle dataset. Neither is
# available in a fresh CI checkout (the dataset can't be committed under Kaggle's terms), so skip
# cleanly there instead of failing on a FileNotFoundError that has nothing to do with the code.
pytestmark = pytest.mark.skipif(
    not model_bundle_dir("public_demo").exists(),
    reason="requires the public-demo model bundle — run `python -m src.train_lgbm --profile public_demo`",
)

VALID_APPLICANT = {
    "age_years": 35,
    "years_employed": 5,
    "income_total": 180_000,
    "credit_amount": 450_000,
    "annuity": 22_500,
    "owns_car": True,
    "owns_realty": True,
    "num_children": 1,
    "family_members": 3,
    "education": "Higher education",
}


@pytest.fixture
def client():
    prediction_rate_limiter._requests.clear()
    with TestClient(app) as c:  # runs the lifespan, so the model actually loads
        yield c
    prediction_rate_limiter._requests.clear()


def test_health_reports_model_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "service_version": "1.3.0",
        "model_version": "1.2.0",
    }


def test_openapi_version_tracks_the_public_demo_release(client: TestClient) -> None:
    assert client.get("/openapi.json").json()["info"]["version"] == "1.3.0"


def test_openapi_describes_payment_difficulty_without_inventing_a_twelve_month_horizon(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()

    assert "payment-difficulty risk" in schema["info"]["description"]
    loss_description = schema["components"]["schemas"]["PredictResponse"]["properties"][
        "expected_credit_loss"
    ]["description"]
    assert "payment-difficulty score" in loss_description
    assert "12-month" not in loss_description


def test_predict_returns_all_expected_fields(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_APPLICANT)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["decision"] in ("approve", "decline")
    assert body["decision_threshold"] == pytest.approx(0.08 / 0.57)
    assert len(body["reason_codes"]) == 3
    assert body["expected_credit_loss"] >= 0
    assert body["lgd_assumption"] == pytest.approx(0.45)
    assert body["model_profile"] == "public_demo"
    assert body["model_name"] == "LightGBMClassifier"
    assert body["model_version"] == "1.2.0"


def test_request_logs_are_allowlisted_for_success_and_rate_limit(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    previous_limit = prediction_rate_limiter.limit
    prediction_rate_limiter.limit = 1
    try:
        assert client.post("/predict", json=VALID_APPLICANT).status_code == 200
        limited = client.post("/predict", json=VALID_APPLICANT)
    finally:
        prediction_rate_limiter.limit = previous_limit

    assert limited.status_code == 429
    records = [
        json.loads(record.message)
        for record in caplog.records
        if '"event": "http_request"' in record.message
    ]
    assert {record["status_code"] for record in records} >= {200, 429}
    for record in records:
        assert set(record) == SAFE_LOG_FIELDS
        assert "income_total" not in json.dumps(record)


def test_predict_decision_matches_threshold(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_APPLICANT)
    body = response.json()

    expected_decision = (
        "decline" if body["probability_of_default"] >= body["decision_threshold"] else "approve"
    )
    assert body["decision"] == expected_decision


def test_predict_riskier_profile_scores_higher_pd(client: TestClient) -> None:
    """Young, no employment history, large loan relative to income vs an established applicant —
    the riskier profile should score a higher PD. A real behavioural check, not just a shape check."""
    safe_applicant = {
        **VALID_APPLICANT,
        "age_years": 45,
        "years_employed": 15,
        "income_total": 400_000,
    }
    risky_applicant = {
        **VALID_APPLICANT,
        "age_years": 21,
        "years_employed": None,
        "income_total": 60_000,
        "credit_amount": 900_000,
    }

    safe_pd = client.post("/predict", json=safe_applicant).json()["probability_of_default"]
    risky_pd = client.post("/predict", json=risky_applicant).json()["probability_of_default"]

    assert risky_pd > safe_pd


def test_predict_ecl_equals_pd_times_lgd_times_credit_amount(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_APPLICANT)
    body = response.json()

    # probability_of_default in the response is rounded to 4dp before this recomputation, so the
    # tolerance has to cover that rounding's worst case: 0.5e-4 x lgd x credit_amount ~= R10
    expected_ecl = (
        body["probability_of_default"] * body["lgd_assumption"] * VALID_APPLICANT["credit_amount"]
    )
    assert body["expected_credit_loss"] == pytest.approx(expected_ecl, abs=15.0)


def test_predict_rejects_missing_required_field(client: TestClient) -> None:
    incomplete = {k: v for k, v in VALID_APPLICANT.items() if k != "income_total"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_rejects_negative_credit_amount(client: TestClient) -> None:
    bad_applicant = {**VALID_APPLICANT, "credit_amount": -1000}
    response = client.post("/predict", json=bad_applicant)
    assert response.status_code == 422


def test_predict_rejects_unknown_education_category(client: TestClient) -> None:
    bad_applicant = {**VALID_APPLICANT, "education": "Made-up category"}
    response = client.post("/predict", json=bad_applicant)
    assert response.status_code == 422


def test_422_body_is_flat_field_message_list_not_nested_loc_dicts(client: TestClient) -> None:
    """Regression guard for the custom validation handler: FastAPI's default 422 body nests each
    error under loc/msg/type/ctx/url, which is correct but makes a caller reconstruct the field
    name from a list. Confirms the flattened 'field: message' format actually ships."""
    bad_applicant = {**VALID_APPLICANT, "credit_amount": -1000}
    response = client.post("/predict", json=bad_applicant)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert all(isinstance(item, str) for item in detail)  # not the default list-of-dicts
    assert any(item.startswith("credit_amount:") for item in detail)


def test_predict_accepts_age_at_lower_boundary(client: TestClient) -> None:
    applicant = {**VALID_APPLICANT, "age_years": 18, "years_employed": 2}
    response = client.post("/predict", json=applicant)
    assert response.status_code == 200


def test_predict_rejects_age_below_lower_boundary(client: TestClient) -> None:
    applicant = {**VALID_APPLICANT, "age_years": 17}
    response = client.post("/predict", json=applicant)
    assert response.status_code == 422


def test_predict_rejects_zero_income(client: TestClient) -> None:
    """income_total uses gt=0, not ge=0 — a loan applicant reporting zero income is a data-entry
    error, not a valid (if unusual) applicant, and should be caught before it reaches the model."""
    applicant = {**VALID_APPLICANT, "income_total": 0}
    response = client.post("/predict", json=applicant)
    assert response.status_code == 422


def test_predict_rejects_income_above_public_contract_limit(client: TestClient) -> None:
    applicant = {**VALID_APPLICANT, "income_total": 5_000_001}

    response = client.post("/predict", json=applicant)

    assert response.status_code == 422


def test_predict_rejects_credit_above_observed_product_limit(client: TestClient) -> None:
    applicant = {**VALID_APPLICANT, "credit_amount": 4_050_001}

    response = client.post("/predict", json=applicant)

    assert response.status_code == 422


def test_predict_accepts_missing_optional_goods_price_and_occupation(client: TestClient) -> None:
    minimal = {
        "age_years": 25,
        "income_total": 120_000,
        "credit_amount": 200_000,
        "annuity": 15_000,
        "num_children": 0,
        "family_members": 1,
        "education": "Secondary / secondary special",
    }
    response = client.post("/predict", json=minimal)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["expected_credit_loss"] >= 0.0


def test_reason_codes_do_not_reference_bureau_fields_absent_from_the_application_model(
    client: TestClient,
) -> None:
    """The served model excludes bureau-only fields, so its reasons must not imply a missing score."""
    response = client.post("/predict", json=VALID_APPLICANT)
    assert response.status_code == 200

    reasons = response.json()["reason_codes"]
    assert not any("credit bureau score" in reason for reason in reasons)


def test_predict_rejects_wrong_type_for_numeric_field(client: TestClient) -> None:
    bad_applicant = {**VALID_APPLICANT, "income_total": "not a number"}
    response = client.post("/predict", json=bad_applicant)
    assert response.status_code == 422


def test_predict_rejects_extra_unknown_fields(client: TestClient) -> None:
    """The request contract must fail closed instead of silently discarding caller data."""
    applicant_with_extra = {**VALID_APPLICANT, "some_future_field": "value"}
    response = client.post("/predict", json=applicant_with_extra)
    assert response.status_code == 422


def test_defaulted_goods_price_stays_numeric_for_lightgbm():
    artifacts = load_artifacts()
    request = ApplicantRequest(
        age_years=25,
        income_total=120_000,
        credit_amount=200_000,
        annuity=15_000,
    )

    row = applicant_to_row(request, artifacts["feature_names"], artifacts["cat_dtypes"])

    assert pd.api.types.is_float_dtype(row["AMT_GOODS_PRICE"])
