"""Plain-language reason codes derived from per-applicant SHAP values."""

import numpy as np
import pandas as pd

FEATURE_DESCRIPTIONS = {
    "NAME_CONTRACT_TYPE": "loan type (cash vs revolving)",
    "EXT_SOURCE_1": "external credit bureau score (source 1)",
    "EXT_SOURCE_2": "external credit bureau score (source 2)",
    "EXT_SOURCE_3": "external credit bureau score (source 3)",
    "AMT_CREDIT": "loan amount",
    "AMT_INCOME_TOTAL": "reported income",
    "AMT_ANNUITY": "monthly loan repayment (annuity)",
    "AMT_GOODS_PRICE": "price of the goods being financed",
    "DAYS_BIRTH": "applicant age",
    "DAYS_EMPLOYED": "length of current employment",
    "DAYS_REGISTRATION": "time since last registration change",
    "DAYS_ID_PUBLISH": "time since ID document was issued",
    "REGION_POPULATION_RELATIVE": "population density of home region",
    "REGION_RATING_CLIENT": "region risk rating",
    "REGION_RATING_CLIENT_W_CITY": "region risk rating (city-adjusted)",
    "NAME_EDUCATION_TYPE": "education level",
    "NAME_INCOME_TYPE": "income type",
    "NAME_FAMILY_STATUS": "family status",
    "OCCUPATION_TYPE": "occupation",
    "ORGANIZATION_TYPE": "employer type",
    "CNT_CHILDREN": "number of children",
    "CNT_FAM_MEMBERS": "family size",
    "OWN_CAR_AGE": "age of owned car",
    "FLAG_OWN_CAR": "car ownership",
    "FLAG_OWN_REALTY": "property ownership",
}


def humanize_feature(name: str) -> str:
    return FEATURE_DESCRIPTIONS.get(name, name.replace("_", " ").lower())


def reason_codes(
    shap_row: pd.Series, feature_row: pd.Series, train_medians: pd.Series, top_n: int = 3
) -> list[str]:
    """Turn the strongest SHAP drivers into short, applicant-facing statements."""
    top_features = shap_row.abs().sort_values(ascending=False).head(top_n).index

    sentences = []
    for feature in top_features:
        shap_value = shap_row[feature]
        value = feature_row[feature]
        description = humanize_feature(feature)

        if pd.isna(value):
            clause = f"missing {description}"
        elif (
            isinstance(value, int | float | np.integer | np.floating)
            and feature in train_medians.index
        ):
            qualifier = "high" if value > train_medians[feature] else "low"
            clause = f"{qualifier} {description}"
        else:
            clause = f"{description} of {value}"

        verb = "raises" if shap_value > 0 else "lowers"
        sentences.append(f"{clause[0].upper()}{clause[1:]} {verb} the estimated default risk.")

    return sentences
