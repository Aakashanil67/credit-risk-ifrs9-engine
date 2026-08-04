import numpy as np
import pandas as pd
import pytest

from src.preprocessing import fit_transform_splits, split_data


@pytest.fixture
def synthetic_applications() -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    n = 500
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(n),
            "TARGET": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
            "AMT_INCOME_TOTAL": rng.normal(150_000, 50_000, size=n),
            "CODE_GENDER": rng.choice(["M", "F"], size=n),
        }
    )


def test_split_indices_do_not_overlap(synthetic_applications: pd.DataFrame) -> None:
    train, val, test = split_data(synthetic_applications)
    train_idx, val_idx, test_idx = set(train.index), set(val.index), set(test.index)

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(synthetic_applications)


def test_split_is_roughly_60_20_20(synthetic_applications: pd.DataFrame) -> None:
    train, val, test = split_data(synthetic_applications)
    n = len(synthetic_applications)

    assert train.shape[0] / n == pytest.approx(0.6, abs=0.02)
    assert val.shape[0] / n == pytest.approx(0.2, abs=0.02)
    assert test.shape[0] / n == pytest.approx(0.2, abs=0.02)


def test_split_preserves_target_stratification(synthetic_applications: pd.DataFrame) -> None:
    train, val, test = split_data(synthetic_applications)
    full_rate = synthetic_applications["TARGET"].mean()

    for split in (train, val, test):
        assert split["TARGET"].mean() == pytest.approx(full_rate, abs=0.03)


def test_imputer_fills_val_with_train_statistic_not_its_own(
    synthetic_applications: pd.DataFrame,
) -> None:
    """The regression this guards against: imputing before splitting, or fitting on val/test.

    Either bug would let validation's own distribution leak into its imputed values. Here train
    and val have deliberately different income distributions, so if the imputer used val's
    median instead of train's, this test catches it.
    """
    train, val, test = split_data(synthetic_applications)
    train = train.copy()
    val = val.copy()

    train.loc[train.index[:5], "AMT_INCOME_TOTAL"] = np.nan
    val.loc[val.index[:5], "AMT_INCOME_TOTAL"] = np.nan
    # give val a wildly different income level so its own median would be an obvious tell
    # (NaN + 900_000 stays NaN, so the 5 rows set to NaN above are still missing)
    val["AMT_INCOME_TOTAL"] = val["AMT_INCOME_TOTAL"] + 900_000

    expected_fill = train["AMT_INCOME_TOTAL"].median()
    train_imputed, val_imputed, _, _ = fit_transform_splits(train, val, test)

    filled_train_rows = train_imputed.loc[train.index[:5], "AMT_INCOME_TOTAL"]
    filled_val_rows = val_imputed.loc[val.index[:5], "AMT_INCOME_TOTAL"]

    assert (filled_train_rows == expected_fill).all()
    assert (filled_val_rows == expected_fill).all()  # not val's own (much higher) median
