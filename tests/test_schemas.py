import pytest
from pydantic import ValidationError

from api.schemas import ApplicantRequest


def valid_request(**overrides) -> ApplicantRequest:
    values = {
        "age_years": 35,
        "years_employed": 5,
        "income_total": 180_000,
        "credit_amount": 450_000,
        "annuity": 22_500,
        "num_children": 1,
        "family_members": 3,
    }
    values.update(overrides)
    return ApplicantRequest(**values)


def test_employment_cannot_precede_working_age() -> None:
    with pytest.raises(ValidationError, match="years_employed"):
        valid_request(age_years=20, years_employed=10)


def test_annuity_cannot_exceed_requested_credit_amount() -> None:
    with pytest.raises(ValidationError, match="annuity"):
        valid_request(credit_amount=10_000, annuity=12_000)


def test_family_members_must_include_children_and_applicant() -> None:
    with pytest.raises(ValidationError, match="family_members"):
        valid_request(num_children=3, family_members=3)


def test_request_rejects_non_finite_amounts() -> None:
    with pytest.raises(ValidationError, match="finite"):
        valid_request(income_total=float("inf"))


def test_goods_price_is_an_optional_collectable_input() -> None:
    request = valid_request(goods_price=300_000)

    assert request.goods_price == 300_000
