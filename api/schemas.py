"""Request/response schemas for POST /predict.

The applicant form asks for the ~20 fields that actually drive the model (see
reports/scorecard.md and reports/figures/shap_bar.png for which ones) rather than all 122 raw
Home Credit columns — anything not asked for is treated as missing, which LightGBM handles
natively since it was trained on genuinely incomplete data.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ApplicantRequest(BaseModel):
    contract_type: Literal["Cash loans", "Revolving loans"] = "Cash loans"
    age_years: float = Field(..., ge=18, le=100, description="Applicant's age in years")
    years_employed: float | None = Field(
        None, ge=0, le=60, description="Years in current employment; omit if not currently employed"
    )
    income_total: float = Field(..., gt=0, description="Annual income, rand")
    credit_amount: float = Field(..., gt=0, description="Requested loan amount, rand")
    annuity: float = Field(..., gt=0, description="Monthly repayment (annuity), rand")
    goods_price: float | None = Field(
        None, gt=0, description="Price of goods financed, if applicable"
    )
    gender: Literal["M", "F"] = Field(..., description="As recorded on the application")
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
    organization_type: str | None = Field(None, description="Employer type, e.g. 'Self-employed'")
    region_population_relative: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Normalised population density of home region; omit if unknown",
    )
    own_car_age: float | None = Field(None, ge=0, le=80)

    model_config = {
        "json_schema_extra": {
            "example": {
                "contract_type": "Cash loans",
                "age_years": 35,
                "years_employed": 5,
                "income_total": 180000,
                "credit_amount": 450000,
                "annuity": 22500,
                "goods_price": 450000,
                "gender": "F",
                "owns_car": True,
                "owns_realty": True,
                "num_children": 1,
                "family_members": 3,
                "education": "Higher education",
                "income_type": "Working",
                "family_status": "Married",
                "occupation": "Core staff",
                "organization_type": "Business Entity Type 3",
            }
        }
    }


class PredictResponse(BaseModel):
    probability_of_default: float
    decision: Literal["approve", "decline"]
    decision_threshold: float
    reason_codes: list[str]
    expected_credit_loss: float = Field(..., description="12-month ECL in rand: PD x LGD x EAD")
    lgd_assumption: float
