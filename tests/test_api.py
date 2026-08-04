import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import LGBM_MODEL_PATH

# These are integration tests against the real trained model, not unit tests — they need
# artifacts produced by `python -m src.train_lgbm`, which needs the Kaggle dataset. Neither is
# available in a fresh CI checkout (the dataset can't be committed under Kaggle's terms), so skip
# cleanly there instead of failing on a FileNotFoundError that has nothing to do with the code.
pytestmark = pytest.mark.skipif(
    not LGBM_MODEL_PATH.exists(),
    reason="requires trained model artifacts — run `python -m src.train_lgbm` first",
)

VALID_APPLICANT = {
    "age_years": 35,
    "years_employed": 5,
    "income_total": 180_000,
    "credit_amount": 450_000,
    "annuity": 22_500,
    "gender": "F",
    "owns_car": True,
    "owns_realty": True,
    "num_children": 1,
    "family_members": 3,
    "education": "Higher education",
}


@pytest.fixture
def client():
    with TestClient(app) as c:  # runs the lifespan, so the model actually loads
        yield c


def test_health_reports_model_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_all_expected_fields(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_APPLICANT)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["decision"] in ("approve", "decline")
    assert body["decision_threshold"] == pytest.approx(0.08)
    assert len(body["reason_codes"]) == 3
    assert body["expected_credit_loss"] >= 0
    assert body["lgd_assumption"] == pytest.approx(0.45)


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


def test_predict_rejects_invalid_gender_literal(client: TestClient) -> None:
    bad_applicant = {**VALID_APPLICANT, "gender": "X"}
    response = client.post("/predict", json=bad_applicant)
    assert response.status_code == 422


def test_422_body_is_flat_field_message_list_not_nested_loc_dicts(client: TestClient) -> None:
    """Regression guard for the custom validation handler: FastAPI's default 422 body nests each
    error under loc/msg/type/ctx/url, which is correct but makes a caller reconstruct the field
    name from a list. Confirms the flattened 'field: message' format actually ships."""
    bad_applicant = {**VALID_APPLICANT, "gender": "X", "credit_amount": -1000}
    response = client.post("/predict", json=bad_applicant)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert all(isinstance(item, str) for item in detail)  # not the default list-of-dicts
    assert any(item.startswith("gender:") for item in detail)
    assert any(item.startswith("credit_amount:") for item in detail)


def test_predict_accepts_age_at_lower_boundary(client: TestClient) -> None:
    applicant = {**VALID_APPLICANT, "age_years": 18}
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


def test_predict_accepts_missing_optional_bureau_and_car_fields(client: TestClient) -> None:
    """No goods_price, own_car_age, region_population_relative, occupation, organization_type —
    the exact shape of a new applicant with no bureau file or car yet."""
    minimal = {
        "age_years": 25,
        "income_total": 120_000,
        "credit_amount": 200_000,
        "annuity": 15_000,
        "gender": "M",
        "num_children": 0,
        "family_members": 1,
        "education": "Secondary / secondary special",
    }
    response = client.post("/predict", json=minimal)
    assert response.status_code == 200
    assert response.json()["probability_of_default"] is not None


def test_predict_rejects_wrong_type_for_numeric_field(client: TestClient) -> None:
    bad_applicant = {**VALID_APPLICANT, "income_total": "not a number"}
    response = client.post("/predict", json=bad_applicant)
    assert response.status_code == 422


def test_predict_rejects_extra_unknown_field_types_gracefully(client: TestClient) -> None:
    """Extra fields Pydantic doesn't know about are ignored by default, not a 500 — confirms the
    schema doesn't accidentally reject well-formed-but-unfamiliar payloads from an older client."""
    applicant_with_extra = {**VALID_APPLICANT, "some_future_field": "value"}
    response = client.post("/predict", json=applicant_with_extra)
    assert response.status_code == 200
