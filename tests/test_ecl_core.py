import pytest

from src.ecl_core import ECLScenario, assign_stage, discounted_scenario_ecl, scenario_ecl_schedule


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

    assert stage_one < 0.45 * 100_000
    assert stage_two > stage_one


def test_survival_weighting_prevents_repeated_default_loss_over_a_long_horizon():
    scenario = ECLScenario("Base", weight=1.0, pd_multiplier=1.0, lgd=1.0)

    loss = discounted_scenario_ecl(
        stage=2,
        pd_annual=0.20,
        ead=100_000,
        annual_eir=0.0,
        remaining_months=60,
        scenario=scenario,
    )

    assert loss == pytest.approx((1 - (1 - 0.20) ** 5) * 100_000)
    assert loss <= 100_000


def test_stage_one_uses_the_annual_pd_once_when_discounting_is_zero():
    scenario = ECLScenario("Base", weight=1.0, pd_multiplier=1.0, lgd=0.45)

    loss = discounted_scenario_ecl(
        stage=1,
        pd_annual=0.20,
        ead=100_000,
        annual_eir=0.0,
        remaining_months=60,
        scenario=scenario,
    )

    assert loss == pytest.approx(0.20 * 0.45 * 100_000)


def test_stage_three_recognises_the_credit_impaired_shortfall_immediately():
    scenario = ECLScenario("Base", weight=1.0, pd_multiplier=1.0, lgd=0.45)

    loss = discounted_scenario_ecl(
        stage=3,
        pd_annual=0.01,
        ead=100_000,
        annual_eir=0.0,
        remaining_months=60,
        scenario=scenario,
    )

    assert loss == pytest.approx(45_000)


def test_scenario_weight_scales_the_expected_loss():
    scenario = ECLScenario("Downside", weight=0.20, pd_multiplier=1.30, lgd=0.55)

    loss = discounted_scenario_ecl(
        stage=1, pd_annual=0.10, ead=100_000, annual_eir=0.15, remaining_months=1, scenario=scenario
    )

    monthly_pd = 1 - (1 - 0.10 * 1.30) ** (1 / 12)
    expected = 0.20 * monthly_pd * 0.55 * 100_000 / 1.15 ** (1 / 12)
    assert loss == pytest.approx(expected)


def test_stage_one_schedule_reconciles_the_monthly_losses_to_the_ecl() -> None:
    scenario = ECLScenario("Base", weight=0.60, pd_multiplier=1.0, lgd=0.45)

    schedule = scenario_ecl_schedule(
        stage=1,
        pd_annual=0.04,
        ead=100_000,
        annual_eir=0.12,
        remaining_months=48,
        scenario=scenario,
    )

    monthly_hazard = 1 - (1 - 0.04) ** (1 / 12)
    assert len(schedule) == 12
    assert schedule[0].survival_at_start == pytest.approx(1.0)
    assert schedule[0].marginal_default_probability == pytest.approx(monthly_hazard)
    assert schedule[0].discount_factor == pytest.approx(1.12 ** (-1 / 12))
    assert sum(period.weighted_loss for period in schedule) == pytest.approx(
        discounted_scenario_ecl(1, 0.04, 100_000, 0.12, 48, scenario)
    )


def test_zero_exposure_produces_zero_loss_in_every_schedule_period() -> None:
    scenario = ECLScenario("Base", weight=1.0, pd_multiplier=1.0, lgd=0.45)

    schedule = scenario_ecl_schedule(1, 0.04, 0.0, 0.12, 48, scenario)

    assert all(period.weighted_loss == 0 for period in schedule)
