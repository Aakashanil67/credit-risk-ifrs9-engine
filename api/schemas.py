"""Request/response schemas for the collectable public-demo prediction contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApplicantRequest(BaseModel):
    contract_type: Literal["Cash loans", "Revolving loans"] = "Cash loans"
    age_years: float = Field(..., ge=18, le=100, description="Applicant's age in years")
    years_employed: float | None = Field(
        None, ge=0, le=60, description="Years in current employment; omit if not currently employed"
    )
    # Credit and goods-price limits are the maximum observed source values. Income is capped below
    # the dataset's isolated 117m outlier so a public form cannot generate meaningless SHAP text.
    income_total: float = Field(
        ...,
        gt=0,
        le=5_000_000,
        description="Annual income in dataset monetary units (up to 5,000,000)",
    )
    credit_amount: float = Field(
        ..., gt=0, le=4_050_000, description="Requested loan amount in dataset monetary units"
    )
    annuity: float = Field(
        ..., gt=0, le=300_000, description="Monthly repayment in dataset monetary units"
    )
    goods_price: float | None = Field(
        None, gt=0, le=4_050_000, description="Price of goods financed, if applicable"
    )
    owns_car: bool = False
    owns_realty: bool = False
    num_children: int = Field(0, ge=0, le=20)
    family_members: int = Field(1, ge=1, le=20)
    education: str = Field(
        "Secondary / secondary special",
        description="e.g. 'Higher education', 'Secondary / secondary special'",
    )
    income_type: str = Field(
        "Working", description="e.g. 'Working', 'Commercial associate', 'Pensioner'"
    )
    family_status: str = Field("Married", description="e.g. 'Married', 'Single / not married'")
    occupation: str | None = Field(None, description="e.g. 'Laborers', 'Sales staff', 'Managers'")
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "contract_type": "Cash loans",
                "age_years": 35,
                "years_employed": 5,
                "income_total": 180000,
                "credit_amount": 450000,
                "annuity": 22500,
                "goods_price": 450000,
                "owns_car": True,
                "owns_realty": True,
                "num_children": 1,
                "family_members": 3,
                "education": "Higher education",
                "income_type": "Working",
                "family_status": "Married",
                "occupation": "Core staff",
            }
        },
    )

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> "ApplicantRequest":
        if self.years_employed is not None and self.years_employed > self.age_years - 14:
            raise ValueError("years_employed cannot exceed years since age 14")
        if self.annuity > self.credit_amount:
            raise ValueError("annuity cannot exceed credit_amount")
        if self.family_members < self.num_children + 1:
            raise ValueError("family_members must include the applicant and every child")
        return self


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    probability_of_default: float = Field(
        ...,
        description=(
            "Estimated probability of the Home Credit payment-difficulty target; the field name "
            "is retained for API compatibility"
        ),
    )
    decision: Literal["approve", "decline"]
    decision_threshold: float
    reason_codes: list[str]
    expected_credit_loss: float = Field(
        ...,
        description=(
            "Illustrative loss in dataset monetary units: payment-difficulty score x assumed LGD "
            "x requested credit; not an IFRS 9 provision"
        ),
    )
    lgd_assumption: float
    expected_value: float = Field(
        ..., description="Illustrative expected value in dataset monetary units"
    )
    model_name: str
    model_version: str
    model_profile: Literal["public_demo"]
