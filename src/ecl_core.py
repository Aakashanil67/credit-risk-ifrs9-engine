"""Discounted, scenario-weighted ECL mechanics for the project demonstration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ECLScenario:
    name: str
    weight: float
    pd_multiplier: float
    lgd: float

    def __post_init__(self) -> None:
        if not 0 <= self.weight <= 1:
            raise ValueError("scenario weight must be between 0 and 1")
        if not 0 <= self.lgd <= 1:
            raise ValueError("scenario LGD must be between 0 and 1")
        if self.pd_multiplier < 0:
            raise ValueError("scenario PD multiplier must be non-negative")


@dataclass(frozen=True)
class ECLPeriod:
    month: int
    survival_at_start: float
    marginal_default_probability: float
    discount_factor: float
    unweighted_loss: float
    weighted_loss: float


def assign_stage(
    pd_at_origination: float,
    pd_current: float,
    days_past_due: int = 0,
    credit_impaired: bool = False,
) -> int:
    """Assign the project's explicit Stage 1/2/3 demonstration rule."""
    if credit_impaired or days_past_due >= 90:
        return 3
    if days_past_due >= 30 or (pd_current >= 0.05 and pd_current >= 2 * pd_at_origination):
        return 2
    return 1


def scenario_ecl_schedule(
    stage: int,
    pd_annual: float,
    ead: float,
    annual_eir: float,
    remaining_months: int,
    scenario: ECLScenario,
) -> tuple[ECLPeriod, ...]:
    """Return the monthly loss schedule for one weighted economic scenario."""
    if stage not in {1, 2, 3}:
        raise ValueError("stage must be 1, 2, or 3")
    if not 0 <= pd_annual <= 1 or ead < 0 or annual_eir < 0 or remaining_months < 1:
        raise ValueError("invalid ECL inputs")

    if stage == 3:
        # A credit-impaired account is already in default for this mechanics demonstration. Its
        # loss is the first discounted cash shortfall, not another stream of annual PD draws.
        discount_factor = (1 + annual_eir) ** (-1 / 12)
        loss = scenario.lgd * ead * discount_factor
        return (
            ECLPeriod(
                month=1,
                survival_at_start=1.0,
                marginal_default_probability=1.0,
                discount_factor=discount_factor,
                unweighted_loss=loss,
                weighted_loss=scenario.weight * loss,
            ),
        )

    horizon = min(12, remaining_months) if stage == 1 else remaining_months
    adjusted_pd = min(pd_annual * scenario.pd_multiplier, 1.0)
    monthly_hazard = 1 - (1 - adjusted_pd) ** (1 / 12)
    survival = 1.0
    periods = []
    for month in range(1, horizon + 1):
        marginal_default_probability = survival * monthly_hazard
        discount_factor = (1 + annual_eir) ** (-month / 12)
        loss = marginal_default_probability * scenario.lgd * ead * discount_factor
        periods.append(
            ECLPeriod(
                month=month,
                survival_at_start=survival,
                marginal_default_probability=marginal_default_probability,
                discount_factor=discount_factor,
                unweighted_loss=loss,
                weighted_loss=scenario.weight * loss,
            )
        )
        survival *= 1 - monthly_hazard
    return tuple(periods)


def discounted_scenario_ecl(
    stage: int,
    pd_annual: float,
    ead: float,
    annual_eir: float,
    remaining_months: int,
    scenario: ECLScenario,
) -> float:
    """Discount monthly expected losses over the Stage 1 or lifetime horizon."""
    return sum(
        period.weighted_loss
        for period in scenario_ecl_schedule(
            stage=stage,
            pd_annual=pd_annual,
            ead=ead,
            annual_eir=annual_eir,
            remaining_months=remaining_months,
            scenario=scenario,
        )
    )
