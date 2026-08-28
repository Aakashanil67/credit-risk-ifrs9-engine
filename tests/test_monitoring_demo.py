import pandas as pd

from src.monitoring_demo import apply_stress, render_monitoring_report


def sample_batch() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100.0] * 20,
            "AMT_CREDIT": [200.0] * 20,
            "OCCUPATION_TYPE": ["Laborers"] * 20,
        }
    )


def test_mild_stress_is_deterministic_and_does_not_mutate_source():
    source = sample_batch()
    first = apply_stress(source, "mild_shift", seed=42)
    second = apply_stress(source, "mild_shift", seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert source["AMT_INCOME_TOTAL"].eq(100).all()
    assert first["AMT_INCOME_TOTAL"].eq(90).all()
    assert first["AMT_CREDIT"].eq(210).all()
    assert first["OCCUPATION_TYPE"].isna().sum() == 2


def test_severe_stress_uses_the_declared_25_percent_missingness():
    stressed = apply_stress(sample_batch(), "severe_shift", seed=42)
    assert stressed["AMT_INCOME_TOTAL"].eq(70).all()
    assert stressed["AMT_CREDIT"].eq(240).all()
    assert stressed["OCCUPATION_TYPE"].isna().sum() == 5


def test_monitoring_report_labels_replay_as_simulated_not_production():
    markdown = render_monitoring_report({"batches": []})
    assert "simulated monitoring" in markdown.lower()
    assert "not production observations" in markdown.lower()
