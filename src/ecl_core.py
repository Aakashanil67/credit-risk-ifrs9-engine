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


def discounted_scenario_ecl(
    stage: int,
    pd_annual: float,
    ead: float,
    annual_eir: float,
    remaining_months: int,
    scenario: ECLScenario,
) -> float:
    """Discount monthly expected losses over the Stage 1 or lifetime horizon."""
    if stage not in {1, 2, 3}:
        raise ValueError("stage must be 1, 2, or 3")
    if not 0 <= pd_annual <= 1 or ead < 0 or annual_eir < 0 or remaining_months < 1:
        raise ValueError("invalid ECL inputs")

    horizon = min(12, remaining_months) if stage == 1 else remaining_months
    adjusted_pd = min(pd_annual * scenario.pd_multiplier, 1.0)
    monthly_pd = 1 - (1 - adjusted_pd) ** (1 / 12)
    discounted_loss = sum(
        monthly_pd * scenario.lgd * ead / (1 + annual_eir) ** (month / 12)
        for month in range(1, horizon + 1)
    )
    return scenario.weight * discounted_loss
