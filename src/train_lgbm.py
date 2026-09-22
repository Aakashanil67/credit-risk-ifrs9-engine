"""Train the public-demo LightGBM risk model and evaluate it against a like-for-like baseline.

Two credit-industry metrics show up alongside AUC because that's what a risk team will actually
ask for:
- Gini = 2*AUC - 1. Same ranking information as AUC, rescaled to [-1, 1] with 0 = random —
  it's the number that ends up in a model-risk committee slide, so it's reported here too.
- KS (Kolmogorov-Smirnov) statistic = the maximum gap between the cumulative distribution of
  scores for defaulters vs non-defaulters. Answers "at the best possible cutoff, how well does
  this model separate the two classes?" — a single number a credit committee can compare against
  their existing scorecard's KS without knowing what AUC means.
"""

import argparse
import hashlib
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.artifacts import ArtifactBundle, save_artifact_bundle
from src.config import (
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COL,
    model_bundle_dir,
)
from src.data_loader import load_application_data
from src.evaluation import binary_metrics
from src.features import build_lgbm_features
from src.model_profiles import ModelProfile
from src.preprocessing import split_data
from src.public_demo_baseline import fit_and_evaluate_public_demo_baseline

PARAM_GRID = [
    {"learning_rate": 0.05, "num_leaves": 31},
    {"learning_rate": 0.05, "num_leaves": 63},
    {"learning_rate": 0.10, "num_leaves": 31},
]
CV_N_ESTIMATORS = (
    300  # fixed and modest during the CV sweep — early stopping picks the real count later
)
CV_FOLDS = 5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def training_metadata(
    profile: ModelProfile,
    model: lgb.LGBMClassifier,
    category_dtypes: dict,
    test_metrics: dict[str, float],
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    data_path: Path,
    selected_params: dict,
    best_iteration: int,
) -> dict:
    return {
        "model_name": "LightGBMClassifier",
        "model_version": "1.2.0",
        "profile": profile.value,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": current_revision(),
        "dataset_sha256": file_sha256(data_path),
        "random_seed": RANDOM_SEED,
        "feature_names": list(model.feature_name_),
        "category_levels": {
            column: [str(value) for value in dtype.categories]
            for column, dtype in category_dtypes.items()
        },
        "split_rows": {"train": train_rows, "validation": validation_rows, "test": test_rows},
        "selection": {
            "parameters": selected_params,
            "early_stopping_iteration": best_iteration,
            "refit_rows": train_rows + validation_rows,
        },
        "test_metrics": test_metrics,
        "package_versions": {
            package: version(package) for package in ("lightgbm", "numpy", "pandas", "scikit-learn")
        },
    }


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


def refit_selected_model(
    train: pd.DataFrame, val: pd.DataFrame, params: dict, best_iteration: int, profile: ModelProfile
) -> tuple[lgb.LGBMClassifier, pd.DataFrame, dict]:
    """Refit the selected tree count on every development row before the one final test score."""
    development = pd.concat([train, val], ignore_index=True)
    development_X = build_lgbm_features(development, profile=profile)
    category_dtypes = {
        column: development_X[column].dtype
        for column in development_X.select_dtypes("category").columns
    }
    model = lgb.LGBMClassifier(
        n_estimators=best_iteration,
        random_state=RANDOM_SEED,
        verbose=-1,
        **params,
    )
    model.fit(development_X, development[TARGET_COL])
    return model, development_X, category_dtypes


def align_category_dtypes(X: pd.DataFrame, category_dtypes: dict) -> pd.DataFrame:
    """Use the fitted category boundaries for validation, test, and serving predictions."""
    aligned = X.copy()
    for column, dtype in category_dtypes.items():
        aligned[column] = aligned[column].astype(dtype)
    return aligned


def score_model(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics = binary_metrics(y_true.to_numpy(), y_pred)
    return {
        "AUC": metrics.auc,
        "Gini": metrics.gini,
        "KS": metrics.ks,
        "Brier": metrics.brier,
        "PR_AUC": metrics.pr_auc,
        "LogLoss": metrics.log_loss,
    }


def train_variant(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    params: dict,
) -> tuple[dict[str, float], int, lgb.LGBMClassifier]:
    """Fit one validation variant and return its metrics, stopping iteration, and fitted model."""
    model = fit_final_model(train_X, train_y, val_X, val_y, params)
    metrics = score_model(val_y, model.predict_proba(val_X)[:, 1])
    return metrics, model.best_iteration_, model


def write_comparison(
    baseline: dict, lgbm: dict, best_params: dict, best_iteration: int, out_path
) -> None:
    lines = [
        "# Public-demo logistic baseline vs LightGBM",
        "",
        "Both models use the same 15-field public-demo contract and the same untouched test "
        "fold. LightGBM parameters come from 5-fold CV on the training fold; early stopping on "
        f"the validation fold selected {best_iteration} trees for `{best_params}`. The final "
        "LightGBM model and logistic baseline were then each fitted on the combined train and "
        "validation folds before this test evaluation.",
        "",
        "| metric | logistic baseline | LightGBM | delta |",
        "|---|---|---|---|",
    ]
    for metric in ("AUC", "Gini", "KS", "Brier", "PR_AUC", "LogLoss"):
        delta = lgbm[metric] - baseline[metric]
        lines.append(f"| {metric} | {baseline[metric]:.4f} | {lgbm[metric]:.4f} | {delta:+.4f} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LightGBM payment-difficulty risk model")
    parser.add_argument(
        "--profile",
        choices=[ModelProfile.PUBLIC_DEMO.value],
        default=ModelProfile.PUBLIC_DEMO.value,
        help="the public-demo contract served by the API and dashboard",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = ModelProfile(args.profile)
    df = load_application_data()
    train, val, test = split_data(df, seed=RANDOM_SEED)

    train_X = build_lgbm_features(train, profile=profile)
    val_X = build_lgbm_features(val, profile=profile)
    test_X = build_lgbm_features(test, profile=profile)
    train_category_dtypes = {
        column: train_X[column].dtype for column in train_X.select_dtypes("category").columns
    }
    val_X = align_category_dtypes(val_X, train_category_dtypes)
    train_y, val_y = train[TARGET_COL], val[TARGET_COL]

    print("5-fold CV over param grid:")
    best_params = cv_select_params(train_X, train_y)
    print(f"selected {best_params}")

    validation_metrics, best_iteration, _selection_model = train_variant(
        train_X, train_y, val_X, val_y, best_params
    )
    print(f"LightGBM validation: {validation_metrics}")

    model, development_X, cat_dtypes = refit_selected_model(
        train, val, best_params, best_iteration, profile
    )
    test_X = align_category_dtypes(test_X, cat_dtypes)

    # The test fold is deliberately absent from CV and early stopping above. At this point the
    # parameters and tree count are frozen, so this is the single final performance estimate.
    test_metrics = score_model(test[TARGET_COL], model.predict_proba(test_X)[:, 1])
    print(f"LightGBM untouched test: {test_metrics}")

    # Persist the exact category boundaries LightGBM was trained on — encoding a category column
    # against a *different* set of categories at inference time silently shifts every code and
    # produces wrong predictions with no error, so inference must reuse these dtypes exactly.
    bundle_dir = model_bundle_dir(profile.value)
    save_artifact_bundle(
        ArtifactBundle(
            model=model,
            category_dtypes=cat_dtypes,
            metadata=training_metadata(
                profile,
                model,
                cat_dtypes,
                test_metrics,
                len(train),
                len(val),
                len(test),
                Path("data/application_train.csv"),
                best_params,
                best_iteration,
            ),
        ),
        bundle_dir,
    )
    print(f"saved tuned {profile.value} bundle to {bundle_dir}")

    baseline_metrics, _baseline_model, _baseline_pred = fit_and_evaluate_public_demo_baseline(
        pd.concat([train, val], ignore_index=True), test
    )
    print(f"public-demo logistic baseline test: {baseline_metrics}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_comparison(
        baseline_metrics,
        test_metrics,
        best_params,
        best_iteration,
        REPORTS_DIR / "model_comparison.md",
    )


if __name__ == "__main__":
    main()
