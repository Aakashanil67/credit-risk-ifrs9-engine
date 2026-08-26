import pandas as pd
import pytest

from src.model_profiles import PUBLIC_DEMO_FEATURES, ModelProfile


def test_public_demo_is_the_only_supported_model_profile() -> None:
    assert list(ModelProfile) == [ModelProfile.PUBLIC_DEMO]


def test_public_demo_profile_uses_the_declared_inputs_and_excludes_gender():
    """Removing a protected input must not silently reintroduce it through feature selection."""
    from src.features import build_lgbm_features

    row = {
        "SK_ID_CURR": 1,
        "TARGET": 0,
        "CODE_GENDER": "F",
        "DAYS_EMPLOYED": -500,
        "AMT_CREDIT": 100_000,
    }
    for feature in PUBLIC_DEMO_FEATURES:
        row.setdefault(feature, "Working")

    features = build_lgbm_features(pd.DataFrame([row]), profile=ModelProfile.PUBLIC_DEMO)

    assert list(features.columns) == PUBLIC_DEMO_FEATURES
    assert "ORGANIZATION_TYPE" not in features
    assert "REGION_POPULATION_RELATIVE" not in features
    assert "OWN_CAR_AGE" not in features


def test_public_demo_profile_names_missing_inputs_instead_of_raising_key_error():
    """A caller needs an actionable contract error, not pandas' internal lookup error."""
    from src.features import build_lgbm_features

    with pytest.raises(ValueError, match="AMT_CREDIT"):
        build_lgbm_features(
            pd.DataFrame([{"DAYS_EMPLOYED": -100}]), profile=ModelProfile.PUBLIC_DEMO
        )


def test_profile_builder_turns_the_employment_sentinel_into_a_missing_value():
    """Home Credit's 365243 employment value is a known sentinel, not a real duration."""
    from src.features import build_lgbm_features

    row = {"SK_ID_CURR": 1, "TARGET": 0, "CODE_GENDER": "F", "DAYS_EMPLOYED": 365243}
    for feature in PUBLIC_DEMO_FEATURES:
        row.setdefault(feature, 1)

    features = build_lgbm_features(pd.DataFrame([row]), profile=ModelProfile.PUBLIC_DEMO)

    assert pd.isna(features.loc[0, "DAYS_EMPLOYED"])


def test_profile_builder_does_not_mutate_the_callers_dataframe():
    """Feature engineering must not replace values in the raw audit data used elsewhere."""
    row = {"SK_ID_CURR": 1, "TARGET": 0, "CODE_GENDER": "F", "DAYS_EMPLOYED": 365243}
    for feature in PUBLIC_DEMO_FEATURES:
        row.setdefault(feature, 1)
    raw = pd.DataFrame([row])

    from src.features import build_lgbm_features

    build_lgbm_features(raw, profile=ModelProfile.PUBLIC_DEMO)

    assert raw.loc[0, "DAYS_EMPLOYED"] == 365243
