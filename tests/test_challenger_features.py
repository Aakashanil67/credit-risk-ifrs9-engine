import numpy as np
import pandas as pd
import pytest

from src.challenger_features import DERIVED_FEATURES, add_collectable_ratios


def test_collectable_ratios_have_declared_business_units_and_do_not_mutate_input():
    source = pd.DataFrame(
        {
            "AMT_CREDIT": [200_000.0],
            "AMT_INCOME_TOTAL": [100_000.0],
            "AMT_ANNUITY": [10_000.0],
            "AMT_GOODS_PRICE": [180_000.0],
            "CNT_FAM_MEMBERS": [2.0],
            "DAYS_EMPLOYED": [-1_000.0],
            "DAYS_BIRTH": [-10_000.0],
        }
    )
    original = source.copy(deep=True)

    result = add_collectable_ratios(source)

    pd.testing.assert_frame_equal(source, original)
    assert list(result.columns[-5:]) == list(DERIVED_FEATURES)
    assert result.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(2.0)
    assert result.loc[0, "ANNUITY_INCOME_RATIO"] == pytest.approx(1.2)
    assert result.loc[0, "GOODS_CREDIT_RATIO"] == pytest.approx(0.9)
    assert result.loc[0, "INCOME_PER_FAMILY_MEMBER"] == pytest.approx(50_000)
    assert result.loc[0, "EMPLOYED_AGE_RATIO"] == pytest.approx(0.1)


def test_zero_denominators_become_missing_not_infinite():
    source = pd.DataFrame(
        {
            "AMT_CREDIT": [0.0],
            "AMT_INCOME_TOTAL": [0.0],
            "AMT_ANNUITY": [1.0],
            "AMT_GOODS_PRICE": [1.0],
            "CNT_FAM_MEMBERS": [0.0],
            "DAYS_EMPLOYED": [-1.0],
            "DAYS_BIRTH": [0.0],
        }
    )

    ratios = add_collectable_ratios(source)[list(DERIVED_FEATURES)].to_numpy()

    assert np.isnan(ratios).all()
