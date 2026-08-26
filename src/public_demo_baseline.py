"""A transparent logistic baseline evaluated on the public-demo feature contract."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_SEED, TARGET_COL
from src.evaluation import binary_metrics
from src.features import build_lgbm_features
from src.model_profiles import ModelProfile


def fit_and_evaluate_public_demo_baseline(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict[str, float], Pipeline, np.ndarray]:
    """Fit a regularised logistic baseline on the same fields and test fold as the served model."""
    train_X = build_lgbm_features(train, profile=ModelProfile.PUBLIC_DEMO)
    test_X = build_lgbm_features(test, profile=ModelProfile.PUBLIC_DEMO)
    numeric = train_X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in train_X.columns if column not in numeric]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(max_iter=1_000, random_state=RANDOM_SEED),
            ),
        ]
    )
    model.fit(train_X, train[TARGET_COL])
    predictions = model.predict_proba(test_X)[:, 1]
    measured = binary_metrics(test[TARGET_COL].to_numpy(), predictions)
    metrics = {
        "AUC": measured.auc,
        "Gini": measured.gini,
        "KS": measured.ks,
        "Brier": measured.brier,
        "PR_AUC": measured.pr_auc,
        "LogLoss": measured.log_loss,
    }
    return metrics, model, predictions
