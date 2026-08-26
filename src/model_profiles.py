"""Feature contracts for the benchmark and application-time PD models."""

from enum import StrEnum


class ModelProfile(StrEnum):
    FULL = "full"
    APPLICATION = "application"


# Kept for offline fairness analysis only. It is never a decision feature.
PROTECTED_AUDIT_COLUMNS = ["CODE_GENDER"]

APPLICATION_FEATURES = [
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
    "ORGANIZATION_TYPE",
    "REGION_POPULATION_RELATIVE",
    "OWN_CAR_AGE",
]
