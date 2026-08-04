from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_application_data, missing_value_report, target_balance


@pytest.fixture
def toy_applications() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 0, 0, 1],
            "AMT_INCOME_TOTAL": [100_000, 150_000, None, 90_000],
            "OCCUPATION_TYPE": [None, None, None, "Laborers"],
            "CODE_GENDER": ["M", "F", "M", "F"],
        }
    )


def test_target_balance_matches_known_ratio(toy_applications: pd.DataFrame) -> None:
    balance = target_balance(toy_applications)
    assert balance["repaid (0)"] == pytest.approx(0.75)
    assert balance["defaulted (1)"] == pytest.approx(0.25)


def test_missing_value_report_excludes_complete_columns(toy_applications: pd.DataFrame) -> None:
    report = missing_value_report(toy_applications)
    assert "CODE_GENDER" not in report.index  # no nulls, shouldn't appear
    assert report.loc["OCCUPATION_TYPE", "missing_count"] == 3
    assert report.loc["AMT_INCOME_TOTAL", "missing_pct"] == 25.0


def test_missing_value_report_sorted_worst_first(toy_applications: pd.DataFrame) -> None:
    report = missing_value_report(toy_applications)
    assert report.index[0] == "OCCUPATION_TYPE"  # 75% missing beats 25%


def test_load_application_data_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download application_train.csv"):
        load_application_data(tmp_path / "does_not_exist.csv")
