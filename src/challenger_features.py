"""Derived features evaluated offline without expanding the public input contract."""

import numpy as np
import pandas as pd

DERIVED_FEATURES = (
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "GOODS_CREDIT_RATIO",
    "INCOME_PER_FAMILY_MEMBER",
    "EMPLOYED_AGE_RATIO",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_collectable_ratios(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ratios derived solely from the public input contract."""
    result = X.copy()
    result["CREDIT_INCOME_RATIO"] = _safe_ratio(result["AMT_CREDIT"], result["AMT_INCOME_TOTAL"])
    result["ANNUITY_INCOME_RATIO"] = _safe_ratio(
        12 * result["AMT_ANNUITY"], result["AMT_INCOME_TOTAL"]
    )
    result["GOODS_CREDIT_RATIO"] = _safe_ratio(result["AMT_GOODS_PRICE"], result["AMT_CREDIT"])
    result["INCOME_PER_FAMILY_MEMBER"] = _safe_ratio(
        result["AMT_INCOME_TOTAL"], result["CNT_FAM_MEMBERS"]
    )
    result["EMPLOYED_AGE_RATIO"] = _safe_ratio(
        result["DAYS_EMPLOYED"].abs(), result["DAYS_BIRTH"].abs()
    )
    return result
