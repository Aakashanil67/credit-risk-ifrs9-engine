import numpy as np
import pandas as pd
import pytest

from src.baseline import engineer_features, score_predictions


def test_logistic_baseline_does_not_construct_a_gender_decision_feature():
    """Gender remains available for offline audits but is not a PD-model input."""
    features = engineer_features(
        pd.DataFrame(
            {
                "AMT_INCOME_TOTAL": [100_000],
                "AMT_CREDIT": [200_000],
                "AMT_ANNUITY": [20_000],
                "AMT_GOODS_PRICE": [180_000],
                "DAYS_BIRTH": [-12_000],
                "DAYS_EMPLOYED": [-1_000],
                "EXT_SOURCE_1": [0.2],
                "EXT_SOURCE_2": [0.3],
                "EXT_SOURCE_3": [0.4],
                "REGION_POPULATION_RELATIVE": [0.01],
                "CNT_CHILDREN": [0],
                "CNT_FAM_MEMBERS": [1],
                "CODE_GENDER": ["F"],
                "FLAG_OWN_CAR": ["N"],
                "FLAG_OWN_REALTY": ["Y"],
            }
        )
    )

    assert "is_male" not in features.columns


def test_baseline_scores_probability_error_alongside_discrimination():
    """The comparison report needs Brier for both challenger and benchmark models."""
    metrics = score_predictions(pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.2, 0.8, 0.9]))

    assert metrics["AUC"] == pytest.approx(1.0)
    assert metrics["Brier"] == pytest.approx(0.025)


def test_baseline_metrics_accept_the_numpy_predictions_returned_by_statsmodels():
    """`Logit.predict` returns an ndarray, so the evaluation boundary must not assume pandas."""
    metrics = score_predictions(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))

    assert metrics["KS"] == pytest.approx(1.0)

