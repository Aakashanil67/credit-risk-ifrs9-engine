import pandas as pd

from src.reason_codes import reason_codes


def test_reason_codes_convert_raw_day_features_to_business_units() -> None:
    shap_row = pd.Series({"DAYS_BIRTH": 0.4, "DAYS_EMPLOYED": -0.2})
    feature_row = pd.Series({"DAYS_BIRTH": -35 * 365.25, "DAYS_EMPLOYED": -5 * 365.25})

    reasons = reason_codes(shap_row, feature_row, pd.Series(dtype=float), top_n=2)

    assert "35 years" in reasons[0]
    assert "5 years" in reasons[1]
    assert not any("high applicant age" in reason for reason in reasons)
    assert not any("low length of current employment" in reason for reason in reasons)


def test_reason_codes_label_amounts_as_dataset_monetary_units() -> None:
    shap_row = pd.Series({"AMT_CREDIT": 0.4})
    feature_row = pd.Series({"AMT_CREDIT": 450_000.0})

    reason = reason_codes(shap_row, feature_row, pd.Series(dtype=float), top_n=1)[0]

    assert "450,000 monetary units" in reason
