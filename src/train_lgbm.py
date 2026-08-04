"""LightGBM on the full feature set, tuned against a small grid, compared to the logistic baseline.

Two credit-industry metrics show up alongside AUC because that's what a risk team will actually
ask for:
- Gini = 2*AUC - 1. Same ranking information as AUC, rescaled to [-1, 1] with 0 = random —
  it's the number that ends up in a model-risk committee slide, so it's reported here too.
- KS (Kolmogorov-Smirnov) statistic = the maximum gap between the cumulative distribution of
  scores for defaulters vs non-defaulters. Answers "at the best possible cutoff, how well does
  this model separate the two classes?" — a single number a credit committee can compare against
  their existing scorecard's KS without knowing what AUC means.
"""

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.baseline import run_and_log as run_baseline
from src.config import (
    LGBM_MODEL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COL,
)
from src.data_loader import load_application_data
from src.preprocessing import split_data

PARAM_GRID = [
    {"learning_rate": 0.05, "num_leaves": 31},
    {"learning_rate": 0.05, "num_leaves": 63},
    {"learning_rate": 0.10, "num_leaves": 31},
]
CV_N_ESTIMATORS = 300  # fixed and modest during the CV sweep — early stopping picks the real count later
CV_FOLDS = 5


def build_lgbm_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ("SK_ID_CURR", TARGET_COL)]
    X = df[feature_cols].copy()
    X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)  # same sentinel bug as baseline
    for col in X.select_dtypes("object").columns:
        X[col] = X[col].astype("category")
    return X


def cv_select_params(X: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    best_params, best_score = None, -np.inf

    for params in PARAM_GRID:
        fold_aucs = []
        for train_idx, holdout_idx in cv.split(X, y):
            model = lgb.LGBMClassifier(
                n_estimators=CV_N_ESTIMATORS,
                random_state=RANDOM_SEED,
                verbose=-1,
                **params,
            )
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict_proba(X.iloc[holdout_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y.iloc[holdout_idx], pred))
        mean_auc = float(np.mean(fold_aucs))
        print(f"  {params} -> mean CV AUC {mean_auc:.4f}")
        if mean_auc > best_score:
            best_score, best_params = mean_auc, params

    return best_params


def fit_final_model(
    train_X: pd.DataFrame, train_y: pd.Series, val_X: pd.DataFrame, val_y: pd.Series, params: dict
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        random_state=RANDOM_SEED,
        verbose=-1,
        **params,
    )
    model.fit(
        train_X,
        train_y,
        eval_set=[(val_X, val_y)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model


def gini(auc: float) -> float:
    return 2 * auc - 1


def ks_statistic(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return ks_2samp(y_pred[y_true == 1], y_pred[y_true == 0]).statistic


def score_model(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_pred)
    return {"AUC": auc, "Gini": gini(auc), "KS": ks_statistic(y_true, y_pred)}


def train_and_log_variant(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    params: dict,
    run_name: str,
) -> tuple[dict[str, float], int, lgb.LGBMClassifier]:
    """Fit one LightGBM variant, log it to MLflow, return its validation metrics, stopping iteration, and the model."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        model = fit_final_model(train_X, train_y, val_X, val_y, params)
        metrics = score_model(val_y, model.predict_proba(val_X)[:, 1])
        mlflow.log_param("best_iteration", model.best_iteration_)
        mlflow.log_metrics({k.lower(): v for k, v in metrics.items()})
        mlflow.lightgbm.log_model(model, "model")
    return metrics, model.best_iteration_, model


def write_comparison(baseline: dict, lgbm: dict, best_params: dict, best_iteration: int, out_path) -> None:
    lines = [
        "# Baseline vs LightGBM",
        "",
        f"LightGBM best params from 5-fold CV: `{best_params}`, stopped at "
        f"{best_iteration} trees via early stopping against validation AUC.",
        "",
        "| metric | logistic baseline | LightGBM | delta |",
        "|---|---|---|---|",
    ]
    for metric in ("AUC", "Gini", "KS"):
        delta = lgbm[metric] - baseline[metric]
        lines.append(
            f"| {metric} | {baseline[metric]:.4f} | {lgbm[metric]:.4f} | {delta:+.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_application_data()
    train, val, _test = split_data(df, seed=RANDOM_SEED)

    train_X = build_lgbm_features(train)
    val_X = build_lgbm_features(val)
    train_y, val_y = train[TARGET_COL], val[TARGET_COL]

    print("5-fold CV over param grid:")
    best_params = cv_select_params(train_X, train_y)
    print(f"selected {best_params}")

    lgbm_metrics, best_iteration, model = train_and_log_variant(
        train_X, train_y, val_X, val_y, best_params, run_name="lgbm_tuned"
    )
    print(f"LightGBM validation: {lgbm_metrics}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LGBM_MODEL_PATH)
    print(f"saved tuned model to {LGBM_MODEL_PATH}")

    baseline_metrics, _model, _pred = run_baseline(train, val)
    print(f"baseline validation: {baseline_metrics}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_comparison(
        baseline_metrics, lgbm_metrics, best_params, best_iteration, REPORTS_DIR / "model_comparison.md"
    )


if __name__ == "__main__":
    main()
