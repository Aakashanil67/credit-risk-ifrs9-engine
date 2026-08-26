import pytest

from src.ecl_core import ECLScenario, assign_stage, discounted_scenario_ecl


def test_stage_assignment_uses_credit_impairment_dpd_and_significant_pd_increase():
    """Stage 3 takes precedence; Stage 2 captures 30+ DPD or a material PD deterioration."""
    assert assign_stage(0.02, 0.03, days_past_due=90) == 3
    assert assign_stage(0.02, 0.05, days_past_due=30) == 2
    assert assign_stage(0.02, 0.05) == 2
    assert assign_stage(0.02, 0.03) == 1


def test_stage_one_limits_the_discounted_loss_horizon_to_twelve_months():
    scenario = ECLScenario("Base", weight=1.0, pd_multiplier=1.0, lgd=0.45)
    stage_one = discounted_scenario_ecl(
        stage=1,
        pd_annual=0.12,
        ead=100_000,
        annual_eir=0.15,
        remaining_months=60,
        scenario=scenario,
    )
    stage_two = discounted_scenario_ecl(
        stage=2,
        pd_annual=0.12,
        ead=100_000,
        annual_eir=0.15,
        remaining_months=60,
        scenario=scenario,
    )

    assert stage_one > 0
    assert stage_two > stage_one


def test_scenario_weight_scales_the_expected_loss():
    scenario = ECLScenario("Downside", weight=0.20, pd_multiplier=1.30, lgd=0.55)

    loss = discounted_scenario_ecl(
        stage=1, pd_annual=0.10, ead=100_000, annual_eir=0.15, remaining_months=1, scenario=scenario
    )

    monthly_pd = 1 - (1 - 0.10 * 1.30) ** (1 / 12)
    expected = 0.20 * monthly_pd * 0.55 * 100_000 / 1.15 ** (1 / 12)
    assert loss == pytest.approx(expected)
