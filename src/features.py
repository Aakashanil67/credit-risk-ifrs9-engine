"""Feature engineering shared by model training, scoring, and offline diagnostics."""

import numpy as np
import pandas as pd

from src.model_profiles import PUBLIC_DEMO_FEATURES, ModelProfile


def build_lgbm_features(df: pd.DataFrame, profile: ModelProfile) -> pd.DataFrame:
    """Build the exact feature matrix declared by a model profile."""
    if profile is not ModelProfile.PUBLIC_DEMO:
        raise ValueError(f"unsupported model profile: {profile}")
    missing = [feature for feature in PUBLIC_DEMO_FEATURES if feature not in df]
    if missing:
        raise ValueError(f"public_demo profile is missing required features: {missing}")
    X = df[PUBLIC_DEMO_FEATURES].copy()
    X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)  # same sentinel bug as baseline
    for col in X.select_dtypes("object").columns:
        X[col] = X[col].astype("category")
    return X
