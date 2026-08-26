"""Write a deterministic, transparent IFRS 9 ECL mechanics demonstration.

The Home Credit data is a cross-sectional application dataset. It has no account-level repayment
history, contractual cash-flow schedules, cure rates, or observed risk migration. This module
therefore demonstrates the accounting mechanics on three fixed examples instead of presenting a
synthetic sample as a real portfolio provision.
"""

from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR
from src.ecl_core import assign_stage
from src.ecl_demo import scenario_weighted_ecl

_MECHANICS_INPUTS = (
    {
        "account": "Performing account",
        "pd_at_origination": 0.03,
        "pd_current": 0.04,
        "days_past_due": 0,
        "credit_impaired": False,
        "ead": 100_000.0,
        "annual_eir": 0.12,
        "remaining_months": 48,
    },
    {
        "account": "SICR account",
        "pd_at_origination": 0.03,
        "pd_current": 0.08,
        "days_past_due": 0,
        "credit_impaired": False,
        "ead": 100_000.0,
        "annual_eir": 0.12,
        "remaining_months": 48,
    },
    {
        "account": "Credit-impaired account",
        "pd_at_origination": 0.03,
        "pd_current": 0.40,
        "days_past_due": 95,
        "credit_impaired": True,
        "ead": 100_000.0,
        "annual_eir": 0.12,
        "remaining_months": 48,
    },
)

_STAGE_NAMES = {
    1: "Stage 1 - performing",
    2: "Stage 2 - significant increase in credit risk",
    3: "Stage 3 - credit-impaired",
}


def build_mechanics_examples() -> pd.DataFrame:
    """Return three reproducible accounts that exercise the documented stage rules."""
    rows = []
    for account in _MECHANICS_INPUTS:
        stage = assign_stage(
            pd_at_origination=account["pd_at_origination"],
            pd_current=account["pd_current"],
            days_past_due=account["days_past_due"],
            credit_impaired=account["credit_impaired"],
        )
        ecl = scenario_weighted_ecl(
            stage=stage,
            pd_annual=account["pd_current"],
            ead=account["ead"],
            annual_eir=account["annual_eir"],
            remaining_months=account["remaining_months"],
        )
        rows.append(
            {
                **account,
                "stage": stage,
                "stage_name": _STAGE_NAMES[stage],
                "ecl": ecl,
                "coverage_pct": 100 * ecl / account["ead"],
            }
        )
    return pd.DataFrame(rows)


def write_ifrs9_summary(examples: pd.DataFrame, out_path: Path) -> None:
    """Write the mechanics report without assigning a currency to source-dataset amounts."""
    lines = [
        "# IFRS 9 expected credit loss mechanics",
        "",
        "This report demonstrates the project's ECL calculation on three fixed accounts. Amounts "
        "are expressed in **dataset monetary units**, because the Home Credit competition data "
        "does not identify a currency. It is not a portfolio provision or a claim about a lender's "
        "actual expected loss.",
        "",
        "## Stage rules used in this demonstration",
        "",
        "- **Stage 1:** 12-month ECL when there is no significant increase in credit risk (SICR).",
        "- **Stage 2:** lifetime ECL when days past due are at least 30, or current PD is at least "
        "double origination PD and at least 5%.",
        "- **Stage 3:** credit-impaired when days past due are at least 90 or the account is flagged "
        "credit-impaired.",
        "",
        "## Scenario-weighted discounted examples",
        "",
        "| account | stage | origination PD | current annual PD | EAD (monetary units) | "
        "remaining months | ECL (monetary units) | coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in examples.itertuples(index=False):
        lines.append(
            f"| {row.account} | {row.stage_name} | {row.pd_at_origination:.1%} | "
            f"{row.pd_current:.1%} | {row.ead:,.0f} | {row.remaining_months} | "
            f"{row.ecl:,.0f} | {row.coverage_pct:.2f}% |"
        )
    lines += [
        "",
        "## Calculation method",
        "",
        "For Stages 1 and 2, annual PD is converted to a constant monthly hazard. Each month's "
        "loss uses the probability that the account has survived to that month and defaults during "
        "that month, multiplied by LGD and EAD, then discounted at the effective interest rate. "
        "Stage 1 is capped at 12 months; Stage 2 runs over the remaining term. The result is "
        "weighted across the stated upside, base, and downside scenarios (20% / 60% / 20%).",
        "",
        "For Stage 3, the demonstration treats the account as already in default and calculates "
        "the first discounted cash shortfall (LGD × EAD), rather than applying another stream of "
        "default probabilities. A production Stage 3 model would project recoveries and costs from "
        "workout cash flows.",
        "",
        "## What this does not model",
        "",
        "The source data does not contain contractual amortisation schedules, account balances over "
        "time, observed transitions between stages, recoveries, forward-looking macroeconomic "
        "variables, or a lender's approved SICR policy. Replacing those assumptions is necessary "
        "before using this method for accounting or credit decisions.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    examples = build_mechanics_examples()
    print(examples[["account", "stage_name", "ead", "ecl", "coverage_pct"]].to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_ifrs9_summary(examples, REPORTS_DIR / "ifrs9_summary.md")


if __name__ == "__main__":
    main()
