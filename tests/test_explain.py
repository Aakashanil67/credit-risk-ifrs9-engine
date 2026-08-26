import numpy as np
import pandas as pd

from src.explain import shap_raw_scores
from src.reason_codes import humanize_feature, reason_codes


def test_reason_codes_picks_largest_public_feature_contributions() -> None:
    shap_row = pd.Series({"DAYS_BIRTH": -0.8, "AMT_CREDIT": 0.05, "CNT_CHILDREN": 0.3})
    feature_row = pd.Series({"DAYS_BIRTH": -35 * 365.25, "AMT_CREDIT": 600_000, "CNT_CHILDREN": 2})

    codes = reason_codes(shap_row, feature_row, top_n=2)

    assert len(codes) == 2
    assert any("35 years" in code for code in codes)
    assert any("children" in code for code in codes)
    assert not any("loan amount" in code for code in codes)


def test_reason_codes_uses_shap_sign_for_public_features() -> None:
    shap_row = pd.Series({"DAYS_BIRTH": -0.9})
    feature_row = pd.Series({"DAYS_BIRTH": -35 * 365.25})

    code = reason_codes(shap_row, feature_row, top_n=1)[0]

    assert "35 years" in code
    assert "lowers" in code


def test_reason_codes_describes_amounts_in_dataset_units() -> None:
    shap_row = pd.Series({"AMT_CREDIT": 0.6})
    feature_row = pd.Series({"AMT_CREDIT": 900_000})

    code = reason_codes(shap_row, feature_row, top_n=1)[0]

    assert "900,000 monetary units" in code
    assert "raises" in code


def test_reason_codes_falls_back_to_value_for_unknown_feature() -> None:
    shap_row = pd.Series({"UNKNOWN_COLUMN": 0.4})
    feature_row = pd.Series({"UNKNOWN_COLUMN": "some_category"})

    code = reason_codes(shap_row, feature_row, top_n=1)[0]

    assert "unknown column of some_category" in code.lower()


def test_reason_codes_flags_missing_public_field() -> None:
    shap_row = pd.Series({"OCCUPATION_TYPE": 0.5})
    feature_row = pd.Series({"OCCUPATION_TYPE": np.nan})

    code = reason_codes(shap_row, feature_row, top_n=1)[0]

    assert "missing occupation" in code.lower()


def test_humanize_feature_uses_public_dictionary_then_falls_back() -> None:
    assert humanize_feature("AMT_CREDIT") == "loan amount"
    assert humanize_feature("SOME_RANDOM_COLUMN") == "some random column"


def test_shap_raw_scores_add_base_value_to_all_feature_contributions() -> None:
    """Tree SHAP explains LightGBM's raw margin, not its post-sigmoid PD."""

    base_values = np.array([-2.0, -2.0])
    values = np.array([[0.2, -0.1], [0.5, 0.4]])

    scores = shap_raw_scores(base_values, values)

    np.testing.assert_allclose(scores, [-1.9, -1.1])
