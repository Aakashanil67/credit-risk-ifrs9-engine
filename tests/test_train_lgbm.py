import pandas as pd

from src.train_lgbm import align_category_dtypes


def test_align_category_dtypes_uses_fitted_category_boundaries() -> None:
    fitted_dtype = pd.CategoricalDtype(categories=["Working", "Pensioner"])
    candidate = pd.DataFrame(
        {"NAME_INCOME_TYPE": pd.Series(["Working", "Unseen category"], dtype="category")}
    )

    aligned = align_category_dtypes(candidate, {"NAME_INCOME_TYPE": fitted_dtype})

    assert aligned["NAME_INCOME_TYPE"].dtype == fitted_dtype
    assert aligned["NAME_INCOME_TYPE"].iloc[0] == "Working"
    assert pd.isna(aligned["NAME_INCOME_TYPE"].iloc[1])
