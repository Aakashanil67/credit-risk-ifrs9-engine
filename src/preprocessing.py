"""Leakage-safe split and imputation.

Split before imputing, always. If you impute first and split second, the median/mode used to
fill a training row was computed partly from validation and test rows — the model has seen a
statistic derived from data it's supposed to be evaluated against. Splitting first and fitting
the imputer on the training fold only keeps validation and test genuinely unseen.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import ID_COL, PROCESSED_DIR, RANDOM_SEED, TARGET_COL, TRAIN_FRACTION, VAL_FRACTION


def split_data(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRACTION,
    val_frac: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 60/20/20 split. Test set stays untouched until final evaluation."""
    train, holdout = train_test_split(
        df, train_size=train_frac, stratify=df[TARGET_COL], random_state=seed
    )
    # holdout is (val_frac + test_frac) of the original; split it evenly between val and test
    val_share_of_holdout = val_frac / (1 - train_frac)
    val, test = train_test_split(
        holdout, train_size=val_share_of_holdout, stratify=holdout[TARGET_COL], random_state=seed
    )
    return train, val, test


def build_imputer(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(exclude="number").columns.tolist()

    imputer = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "categorical",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent"))]),
                categorical_cols,
            ),
        ],
        verbose_feature_names_out=False,
    )
    imputer.set_output(transform="pandas")
    return imputer, numeric_cols, categorical_cols


def fit_transform_splits(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Fit the imputer on train only; val and test are transform-only, never fit."""
    imputer, numeric_cols, categorical_cols = build_imputer(train)

    train_imputed = imputer.fit_transform(train)
    val_imputed = imputer.transform(val)
    test_imputed = imputer.transform(test)

    for split, imputed in [(train, train_imputed), (val, val_imputed), (test, test_imputed)]:
        imputed.index = split.index
        imputed[ID_COL] = split[ID_COL]
        imputed[TARGET_COL] = split[TARGET_COL]

    return train_imputed, val_imputed, test_imputed, imputer


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED_DIR / "train.parquet")
    val.to_parquet(PROCESSED_DIR / "val.parquet")
    test.to_parquet(PROCESSED_DIR / "test.parquet")


def main() -> None:
    from src.data_loader import load_application_data

    df = load_application_data()
    train, val, test = split_data(df)
    print(f"train={len(train):,} val={len(val):,} test={len(test):,}")

    train_imputed, val_imputed, test_imputed, _ = fit_transform_splits(train, val, test)
    save_splits(train_imputed, val_imputed, test_imputed)
    print(f"saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
