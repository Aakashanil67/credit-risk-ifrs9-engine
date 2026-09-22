"""Feature contract for the deployed public-demo risk model."""

from enum import StrEnum


class ModelProfile(StrEnum):
    PUBLIC_DEMO = "public_demo"


# These are the fields the public dashboard can collect without inventing a value for a feature
# that was complete in Home Credit's training table. Organisation type and regional density are
# dataset-specific, and a visitor cannot reasonably supply either one.
PUBLIC_DEMO_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE",
]
