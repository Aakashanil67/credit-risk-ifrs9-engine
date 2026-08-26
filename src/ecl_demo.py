"""Project-level economic scenarios for the IFRS 9 demonstration."""

from src.ecl_core import ECLScenario, discounted_scenario_ecl

# These weights and loss assumptions are illustrative. They are deliberately kept here rather
# than buried in the dashboard so a reviewer can see exactly what drives the displayed ECL.
DEFAULT_SCENARIOS = (
    ECLScenario("Upside", weight=0.20, pd_multiplier=0.80, lgd=0.40),
    ECLScenario("Base", weight=0.60, pd_multiplier=1.00, lgd=0.45),
    ECLScenario("Downside", weight=0.20, pd_multiplier=1.30, lgd=0.55),
)


def scenario_weighted_ecl(
    stage: int,
    pd_annual: float,
    ead: float,
    annual_eir: float,
    remaining_months: int,
) -> float:
    """Return the probability-weighted discounted ECL across the three scenarios."""
    return sum(
        discounted_scenario_ecl(
            stage=stage,
            pd_annual=pd_annual,
            ead=ead,
            annual_eir=annual_eir,
            remaining_months=remaining_months,
            scenario=scenario,
        )
        for scenario in DEFAULT_SCENARIOS
    )
