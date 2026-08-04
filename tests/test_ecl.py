import pandas as pd
import pytest

from src.ecl import (
    approximate_loan_term_years,
    compute_ecl,
    lifetime_pd_from_12m,
    portfolio_summary,
    stage_loans,
)


def test_lifetime_pd_compounds_above_12m_pd() -> None:
    pd_12m = pd.Series([0.1, 0.1])
    term_years = pd.Series([1.0, 3.0])

    lifetime = lifetime_pd_from_12m(pd_12m, term_years)

    assert lifetime.iloc[0] == pytest.approx(0.1)  # 1-year term: lifetime == 12-month PD
    assert lifetime.iloc[1] == pytest.approx(1 - 0.9**3)  # 3-year term: compounds up
    assert lifetime.iloc[1] > lifetime.iloc[0]


def test_approximate_loan_term_clips_extreme_annuities() -> None:
    df = pd.DataFrame(
        {"AMT_CREDIT": [100_000, 100_000, 100_000], "AMT_ANNUITY": [50_000, 1, 5_000]}
    )
    term = approximate_loan_term_years(df)

    assert term.iloc[0] == pytest.approx(12 / 12)  # 2-month term clipped up to the 12-month floor
    assert term.iloc[1] == pytest.approx(360 / 12)  # absurd 100,000-month term clipped to 30 years
    assert term.iloc[2] == pytest.approx(20 / 12)  # 100_000/5_000 = 20 months, within bounds


def test_stage_loans_assigns_stage_1_when_no_sicr_and_low_pd() -> None:
    current_pd = pd.Series([0.05])
    origination_pd = pd.Series([0.04])  # ratio 1.25, below the 2x SICR threshold

    stage = stage_loans(current_pd, origination_pd)

    assert stage.iloc[0] == 1


def test_stage_loans_assigns_stage_2_when_pd_doubles() -> None:
    current_pd = pd.Series([0.10])
    origination_pd = pd.Series(
        [0.04]
    )  # ratio 2.5x >= threshold, but still below 0.5 impaired cutoff

    stage = stage_loans(current_pd, origination_pd)

    assert stage.iloc[0] == 2


def test_stage_loans_assigns_stage_3_when_credit_impaired_regardless_of_ratio() -> None:
    current_pd = pd.Series([0.6])
    origination_pd = pd.Series([0.55])  # ratio only ~1.1x, would be Stage 1 on ratio alone

    stage = stage_loans(current_pd, origination_pd)

    assert stage.iloc[0] == 3  # PD itself exceeds the impaired threshold


def test_stage_loans_handles_zero_origination_pd_without_dividing_by_zero() -> None:
    current_pd = pd.Series([0.05])
    origination_pd = pd.Series([0.0])

    stage = stage_loans(current_pd, origination_pd)  # must not raise or produce inf/NaN

    assert stage.iloc[0] == 2  # any positive current PD against ~0 origination PD is a huge ratio


def test_compute_ecl_stage_1_uses_12m_pd_not_lifetime() -> None:
    df = pd.DataFrame({"SK_ID_CURR": [1], "AMT_CREDIT": [100_000], "AMT_ANNUITY": [10_000]})
    current_pd = pd.Series([0.05], index=df.index)
    origination_pd = pd.Series([0.045], index=df.index)  # ratio ~1.1, stays Stage 1

    ecl_df = compute_ecl(df, current_pd, origination_pd, lgd=0.5)

    assert ecl_df["stage"].iloc[0] == 1
    expected_ecl = 0.05 * 0.5 * 100_000  # PD_12m x LGD x EAD, not the compounded lifetime PD
    assert ecl_df["ECL"].iloc[0] == pytest.approx(expected_ecl)


def test_compute_ecl_stage_2_uses_lifetime_pd_which_exceeds_12m_ecl() -> None:
    # AMT_CREDIT/AMT_ANNUITY = 50 months (~4.2 years) so lifetime PD genuinely compounds past
    # the 12-month figure — a 1-year-term loan would make lifetime == 12m PD exactly, hiding the bug.
    df = pd.DataFrame({"SK_ID_CURR": [1], "AMT_CREDIT": [100_000], "AMT_ANNUITY": [2_000]})
    current_pd = pd.Series([0.10], index=df.index)
    origination_pd = pd.Series([0.04], index=df.index)  # ratio 2.5x -> SICR -> Stage 2

    ecl_df = compute_ecl(df, current_pd, origination_pd, lgd=0.5)

    ecl_12m_equivalent = 0.10 * 0.5 * 100_000
    assert ecl_df["stage"].iloc[0] == 2
    assert ecl_df["ECL"].iloc[0] > ecl_12m_equivalent  # lifetime provisioning is strictly larger


def test_portfolio_summary_totals_match_sum_of_rows() -> None:
    ecl_df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "EAD": [100_000, 200_000, 300_000],
            "ECL": [5_000, 20_000, 45_000],
            "stage": [1, 2, 3],
        }
    )

    summary = portfolio_summary(ecl_df)

    assert summary["total_ead"] == 600_000
    assert summary["total_ecl"] == 70_000
    assert summary["coverage_pct"] == pytest.approx(70_000 / 600_000 * 100)
    assert summary["by_stage"].loc[1, "loans"] == 1
    assert summary["by_stage"]["ecl"].sum() == 70_000
