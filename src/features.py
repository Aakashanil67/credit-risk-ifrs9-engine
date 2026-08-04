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

from src.config import TARGET_COL


def build_lgbm_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ("SK_ID_CURR", TARGET_COL)]
    X = df[feature_cols].copy()
    X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)  # same sentinel bug as baseline
    for col in X.select_dtypes("object").columns:
        X[col] = X[col].astype("category")
    return X
