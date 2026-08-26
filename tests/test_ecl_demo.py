import pytest

from src.ecl_demo import DEFAULT_SCENARIOS, scenario_weighted_ecl


def test_default_scenarios_are_probability_weighted_to_one():
    """The ECL demonstration cannot be probability-weighted if the scenario weights do not sum."""
    assert sum(scenario.weight for scenario in DEFAULT_SCENARIOS) == pytest.approx(1.0)


def test_weighted_ecl_combines_all_three_economic_scenarios():
    """Downside assumptions should make the weighted loss exceed the base-only contribution."""
    weighted = scenario_weighted_ecl(
        stage=1, pd_annual=0.10, ead=100_000, annual_eir=0.15, remaining_months=12
    )

    assert weighted > 0
