"""Plain-language reason codes derived from per-applicant SHAP values."""

import numpy as np
import pandas as pd

FEATURE_DESCRIPTIONS = {
    "NAME_CONTRACT_TYPE": "loan type (cash vs revolving)",
    "AMT_CREDIT": "loan amount",
    "AMT_INCOME_TOTAL": "reported income",
    "AMT_ANNUITY": "monthly loan repayment (annuity)",
    "AMT_GOODS_PRICE": "price of the goods being financed",
    "DAYS_BIRTH": "applicant age",
    "DAYS_EMPLOYED": "length of current employment",
    "NAME_EDUCATION_TYPE": "education level",
    "NAME_INCOME_TYPE": "income type",
    "NAME_FAMILY_STATUS": "family status",
    "OCCUPATION_TYPE": "occupation",
    "CNT_CHILDREN": "number of children",
    "CNT_FAM_MEMBERS": "family size",
    "FLAG_OWN_CAR": "car ownership",
    "FLAG_OWN_REALTY": "property ownership",
}


def humanize_feature(name: str) -> str:
    return FEATURE_DESCRIPTIONS.get(name, name.replace("_", " ").lower())


def _numeric_clause(feature: str, value: float, description: str) -> str:
    """Format model-scale numeric inputs in units a dashboard visitor can interpret."""
    if feature == "DAYS_BIRTH":
        years = round(-value / 365.25)
        return f"applicant age ({years} {'year' if years == 1 else 'years'})"
    if feature == "DAYS_EMPLOYED":
        years = round(-value / 365.25)
        return f"length of current employment ({years} {'year' if years == 1 else 'years'})"
    if feature.startswith("AMT_"):
        return f"{description} ({value:,.0f} monetary units)"
    if feature in {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}:
        return f"{description} ({value:.0f})"
    return f"{description} ({value:.3g})"


def reason_codes(shap_row: pd.Series, feature_row: pd.Series, top_n: int = 3) -> list[str]:
    """Turn the strongest SHAP drivers into short, applicant-facing statements."""
    top_features = shap_row.abs().sort_values(ascending=False).head(top_n).index

    sentences = []
    for feature in top_features:
        shap_value = shap_row[feature]
        value = feature_row[feature]
        description = humanize_feature(feature)

        if pd.isna(value):
            clause = f"missing {description}"
        elif isinstance(value, int | float | np.integer | np.floating):
            clause = _numeric_clause(feature, float(value), description)
        else:
            clause = f"{description} of {value}"

        verb = "raises" if shap_value > 0 else "lowers"
        sentences.append(
            f"{clause[0].upper()}{clause[1:]} {verb} the estimated payment-difficulty risk."
        )

    return sentences
