"""Feature engineering shared between training and serving.

Split out of src/train_lgbm.py so the serving path (api/scoring.py -> src/explain.py) can use
build_lgbm_features() without transitively importing src/train_lgbm.py -> src/baseline.py, which
pulls in statsmodels and mlflow — training-only packages with no business being in a deploy
target's dependency graph. This split is what let Streamlit Community Cloud's build succeed at
all: with the untrimmed import chain, `pip`/`uv` had to resolve optbinning's ortools requirement
(which needs numpy>=2.0.2) against this project's numpy==1.26.4 pin, which is unsatisfiable.
"""

import numpy as np
import pandas as pd

from src.config import ID_COL, TARGET_COL
from src.model_profiles import APPLICATION_FEATURES, PROTECTED_AUDIT_COLUMNS, ModelProfile


def build_lgbm_features(df: pd.DataFrame, profile: ModelProfile) -> pd.DataFrame:
    """Build the exact feature matrix declared by a model profile."""
    if profile is ModelProfile.APPLICATION:
        missing = [feature for feature in APPLICATION_FEATURES if feature not in df]
        if missing:
            raise ValueError(f"Application profile is missing required features: {missing}")
        X = df[APPLICATION_FEATURES].copy()
    elif profile is ModelProfile.FULL:
        excluded = {ID_COL, TARGET_COL, *PROTECTED_AUDIT_COLUMNS}
        X = df[[column for column in df.columns if column not in excluded]].copy()
    else:
        raise ValueError(f"Unsupported feature profile: {profile}")
    X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)  # same sentinel bug as baseline
    for col in X.select_dtypes("object").columns:
        X[col] = X[col].astype("category")
    return X
