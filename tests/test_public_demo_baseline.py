import pandas as pd

from src.public_demo_baseline import fit_and_evaluate_public_demo_baseline


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(1, 9),
            "TARGET": [0, 1, 0, 1, 0, 1, 0, 1],
            "NAME_CONTRACT_TYPE": ["Cash loans"] * 8,
            "DAYS_BIRTH": [-15_000, -8_000, -16_000, -7_500, -17_000, -7_000, -18_000, -6_500],
            "DAYS_EMPLOYED": [-2_000, -100, -3_000, -50, -4_000, -20, -5_000, -10],
            "AMT_INCOME_TOTAL": [200_000, 60_000, 220_000, 55_000, 250_000, 50_000, 300_000, 45_000],
            "AMT_CREDIT": [100_000, 400_000, 120_000, 450_000, 150_000, 500_000, 180_000, 550_000],
            "AMT_ANNUITY": [10_000, 30_000, 12_000, 32_000, 14_000, 34_000, 16_000, 36_000],
            "AMT_GOODS_PRICE": [100_000, 400_000, 120_000, 450_000, 150_000, 500_000, 180_000, 550_000],
            "FLAG_OWN_CAR": ["Y", "N", "Y", "N", "Y", "N", "Y", "N"],
            "FLAG_OWN_REALTY": ["Y", "N", "Y", "N", "Y", "N", "Y", "N"],
            "CNT_CHILDREN": [0, 1, 0, 2, 0, 1, 0, 2],
            "CNT_FAM_MEMBERS": [1, 3, 1, 4, 1, 3, 1, 4],
            "NAME_EDUCATION_TYPE": ["Higher education", "Secondary / secondary special"] * 4,
            "NAME_INCOME_TYPE": ["Working"] * 8,
            "NAME_FAMILY_STATUS": ["Married"] * 8,
            "OCCUPATION_TYPE": ["Managers", "Laborers"] * 4,
        }
    )


def test_public_demo_logistic_baseline_uses_the_served_contract() -> None:
    train, test = _rows().iloc[:6], _rows().iloc[6:]

    metrics, _model, predictions = fit_and_evaluate_public_demo_baseline(train, test)

    assert set(metrics) == {"AUC", "Gini", "KS", "Brier", "PR_AUC", "LogLoss"}
    assert len(predictions) == len(test)
    assert (predictions >= 0).all() and (predictions <= 1).all()
