import numpy as np
import pandas as pd
import pytest

from src.explain import humanize_feature, reason_codes


@pytest.fixture
def train_medians() -> pd.Series:
    return pd.Series({"EXT_SOURCE_3": 0.5, "AMT_CREDIT": 500_000, "CNT_CHILDREN": 0})


def test_reason_codes_picks_top_n_by_absolute_shap(train_medians: pd.Series) -> None:
    shap_row = pd.Series({"EXT_SOURCE_3": -0.8, "AMT_CREDIT": 0.05, "CNT_CHILDREN": 0.3})
    feature_row = pd.Series({"EXT_SOURCE_3": 0.1, "AMT_CREDIT": 600_000, "CNT_CHILDREN": 2})

    codes = reason_codes(shap_row, feature_row, train_medians, top_n=2)

    assert len(codes) == 2
    # EXT_SOURCE_3 (|shap|=0.8) and CNT_CHILDREN (|shap|=0.3) beat AMT_CREDIT (|shap|=0.05)
    assert any("credit bureau score" in c for c in codes)
    assert any("children" in c for c in codes)
    assert not any("loan amount" in c for c in codes)


def test_reason_codes_direction_matches_shap_sign(train_medians: pd.Series) -> None:
    shap_row = pd.Series({"EXT_SOURCE_3": -0.9})
    feature_row = pd.Series({"EXT_SOURCE_3": 0.1})  # below train median of 0.5 -> "low"

    codes = reason_codes(shap_row, feature_row, train_medians, top_n=1)

    assert len(codes) == 1
    assert "low external credit bureau score" in codes[0].lower()
    assert "lowers" in codes[0]  # negative SHAP -> lowers risk


def test_reason_codes_high_qualifier_and_raises_risk(train_medians: pd.Series) -> None:
    shap_row = pd.Series({"AMT_CREDIT": 0.6})
    feature_row = pd.Series({"AMT_CREDIT": 900_000})  # above train median of 500,000 -> "high"

    codes = reason_codes(shap_row, feature_row, train_medians, top_n=1)

    assert "high loan amount" in codes[0].lower()
    assert "raises" in codes[0]


def test_reason_codes_falls_back_to_value_for_unknown_medians(train_medians: pd.Series) -> None:
    shap_row = pd.Series({"UNKNOWN_COLUMN": 0.4})
    feature_row = pd.Series({"UNKNOWN_COLUMN": "some_category"})

    codes = reason_codes(shap_row, feature_row, train_medians, top_n=1)

    assert "unknown column of some_category" in codes[0].lower()


def test_reason_codes_flags_missing_value_instead_of_calling_it_low(
    train_medians: pd.Series,
) -> None:
    """A new applicant with no bureau file yet has NaN EXT_SOURCE_3, not a genuinely low score —
    saying 'low' would be a false, and consequential, claim about a real person's credit file."""
    shap_row = pd.Series({"EXT_SOURCE_3": 0.5})
    feature_row = pd.Series({"EXT_SOURCE_3": np.nan})

    codes = reason_codes(shap_row, feature_row, train_medians, top_n=1)

    assert "missing external credit bureau score" in codes[0].lower()
    assert "low" not in codes[0].lower()


def test_humanize_feature_uses_dictionary_then_falls_back() -> None:
    assert humanize_feature("EXT_SOURCE_2") == "external credit bureau score (source 2)"
    assert humanize_feature("SOME_RANDOM_COLUMN") == "some random column"
