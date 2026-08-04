"""IFRS 9 expected credit loss: ECL = PD x LGD x EAD, staged into 12-month vs lifetime.

- PD (probability of default): how likely the borrower is to default.
- LGD (loss given default): the fraction of exposure the lender doesn't recover if they do.
  Configurable per call; defaults to 45%, a standard unsecured-consumer-lending assumption.
- EAD (exposure at default): how much is actually outstanding at the point of default. Here,
  the original credit amount (`AMT_CREDIT`) — a simplification; a live book would track the
  amortised outstanding balance instead.
- ECL: the rand amount a lender must set aside today against expected future losses.

IFRS 9 staging (this is the standard's core mechanic, not a modelling choice):
- Stage 1 — performing, no significant increase in credit risk (SICR) since origination.
  Provision for 12-month ECL only.
- Stage 2 — SICR has happened but the loan isn't yet credit-impaired. Provision for lifetime ECL
  (the loss expected over the whole remaining term, not just the next 12 months) — the SICR
  trigger is what forces the jump from a 12-month to a lifetime provision.
- Stage 3 — credit-impaired (the IFRS 9 equivalent of "in default"). Lifetime ECL, same as
  Stage 2, but the loan is now treated as impaired for disclosure purposes.

SICR proxy: this dataset is a single cross-sectional snapshot — there's no repeat observation of
the same loan's PD over time to check "has PD doubled since origination" literally. As a
documented proxy, origination-time risk is read off the logistic baseline (the simpler model a
lender might plausibly have used at underwriting) and current risk off the tuned LightGBM model
(a more powerful model run at monitoring time). SICR fires when LightGBM's PD is at least
SICR_PD_RATIO_THRESHOLD times the baseline's PD for the same applicant. This is a genuine
limitation of building this off Kaggle's static competition data, not a claim of real time-series
monitoring — see reports/ifrs9_summary.md.
"""

import pandas as pd

from src.baseline import run_and_log as run_baseline
from src.config import (
    DEFAULT_EAD_COL,
    DEFAULT_LGD,
    RANDOM_SEED,
    REPORTS_DIR,
    SICR_PD_RATIO_THRESHOLD,
)
from src.data_loader import load_application_data
from src.explain import load_or_train_model
from src.preprocessing import split_data
from src.train_lgbm import build_lgbm_features

CREDIT_IMPAIRED_PD_THRESHOLD = 0.5  # Stage 3 if the model gives default better than even odds
MIN_TERM_MONTHS = 12
MAX_TERM_MONTHS = (
    360  # 30 years — bounds an occasional AMT_ANNUITY near zero from blowing up the term
)


def approximate_loan_term_years(df: pd.DataFrame) -> pd.Series:
    """AMT_CREDIT / AMT_ANNUITY approximates months-to-repay, ignoring interest — a real
    amortisation schedule would use the contractual term, which isn't in this dataset."""
    term_months = (df["AMT_CREDIT"] / df["AMT_ANNUITY"]).clip(
        lower=MIN_TERM_MONTHS, upper=MAX_TERM_MONTHS
    )
    return term_months / 12


def lifetime_pd_from_12m(pd_12m: pd.Series, term_years: pd.Series) -> pd.Series:
    """Compounds a 12-month PD into a cumulative lifetime PD assuming a constant annual hazard
    rate: 1 - (1 - PD_12m)^term_years. Standard technique when only a single PD estimate exists."""
    return 1 - (1 - pd_12m) ** term_years


def stage_loans(
    current_pd: pd.Series,
    origination_pd: pd.Series,
    sicr_ratio: float = SICR_PD_RATIO_THRESHOLD,
    impaired_threshold: float = CREDIT_IMPAIRED_PD_THRESHOLD,
) -> pd.Series:
    ratio = current_pd / origination_pd.clip(lower=1e-6)
    stage = pd.Series(1, index=current_pd.index, name="stage")
    stage[ratio >= sicr_ratio] = 2
    stage[current_pd >= impaired_threshold] = 3
    return stage


def compute_ecl(
    df: pd.DataFrame,
    current_pd: pd.Series,
    origination_pd: pd.Series,
    lgd: float = DEFAULT_LGD,
    ead_col: str = DEFAULT_EAD_COL,
) -> pd.DataFrame:
    ead = df[ead_col]
    term_years = approximate_loan_term_years(df)
    lifetime_pd = lifetime_pd_from_12m(current_pd, term_years)
    stage = stage_loans(current_pd, origination_pd)

    ecl_12m = current_pd * lgd * ead
    ecl_lifetime = lifetime_pd * lgd * ead
    ecl = ecl_12m.where(stage == 1, ecl_lifetime)  # Stage 1 -> 12m ECL, Stage 2/3 -> lifetime ECL

    return pd.DataFrame(
        {
            "SK_ID_CURR": df["SK_ID_CURR"],
            "EAD": ead,
            "PD_origination": origination_pd,
            "PD_12m": current_pd,
            "PD_lifetime": lifetime_pd,
            "term_years": term_years,
            "stage": stage,
            "ECL": ecl,
        }
    )


def portfolio_summary(ecl_df: pd.DataFrame) -> dict:
    total_ead = ecl_df["EAD"].sum()
    total_ecl = ecl_df["ECL"].sum()
    by_stage = ecl_df.groupby("stage").agg(
        loans=("SK_ID_CURR", "count"), ead=("EAD", "sum"), ecl=("ECL", "sum")
    )
    by_stage["coverage_pct"] = (by_stage["ecl"] / by_stage["ead"] * 100).round(3)
    return {
        "total_ead": total_ead,
        "total_ecl": total_ecl,
        "coverage_pct": total_ecl / total_ead * 100,
        "by_stage": by_stage,
    }


def write_ifrs9_summary(summary: dict, lgd: float, out_path) -> None:
    by_stage = summary["by_stage"]
    lines = [
        "# IFRS 9 expected credit loss — portfolio summary",
        "",
        f"LGD assumption: {lgd:.0%} (loss given default — the share of exposure not recovered "
        "after a default). EAD: outstanding credit amount at application (`AMT_CREDIT`).",
        "",
        f"**Total exposure (EAD): R{summary['total_ead']:,.0f}**",
        f"**Total provision (ECL): R{summary['total_ecl']:,.0f}**",
        f"**Portfolio coverage ratio: {summary['coverage_pct']:.3f}%** of exposure held as provision.",
        "",
        "## By IFRS 9 stage",
        "",
        "| stage | loans | EAD (R) | ECL (R) | coverage % |",
        "|---|---|---|---|---|",
    ]
    stage_names = {1: "1 — performing", 2: "2 — SICR (lifetime ECL)", 3: "3 — credit-impaired"}
    for stage, row in by_stage.iterrows():
        lines.append(
            f"| {stage_names.get(stage, stage)} | {row['loans']:,.0f} | {row['ead']:,.0f} | "
            f"{row['ecl']:,.0f} | {row['coverage_pct']:.3f}% |"
        )
    lines += [
        "",
        "## Reading this in IFRS 9 language",
        "",
        "- **Stage 1** loans get 12-month ECL — the expected loss from default events plausible "
        "in the next 12 months.",
        "- **Stage 2** loans have shown a significant increase in credit risk (SICR) since "
        "origination and move to lifetime ECL — expected loss over the full remaining term, "
        "which is materially larger per loan than a 12-month provision.",
        "- **Stage 3** loans are credit-impaired (PD >= 50%) — also lifetime ECL, and flagged "
        "separately for disclosure as IFRS 9 requires.",
        "",
        "**Limitation, stated plainly**: SICR is proxied by comparing the tuned LightGBM model's "
        'PD ("current") against the logistic baseline\'s PD ("origination") for the same '
        "applicant, because this dataset is one static snapshot with no repeat observations of "
        "the same loan over time. A production system would compare a loan's PD today against "
        "its own PD at the actual origination date, not two different models' opinions of the "
        "same application. Treat the stage split here as illustrative of the mechanics, not a "
        "real portfolio's risk migration.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_application_data()
    train, val, _test = split_data(df, seed=RANDOM_SEED)

    lgbm_model = load_or_train_model()
    val_X_lgbm = build_lgbm_features(val)
    current_pd = pd.Series(lgbm_model.predict_proba(val_X_lgbm)[:, 1], index=val_X_lgbm.index)

    _baseline_metrics, _baseline_model, baseline_val_pred = run_baseline(train, val)
    origination_pd = pd.Series(baseline_val_pred, index=val.index)

    ecl_df = compute_ecl(val, current_pd, origination_pd)
    summary = portfolio_summary(ecl_df)

    print(f"total EAD: R{summary['total_ead']:,.0f}")
    print(f"total ECL: R{summary['total_ecl']:,.0f}")
    print(f"coverage ratio: {summary['coverage_pct']:.3f}%")
    print(summary["by_stage"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_ifrs9_summary(summary, DEFAULT_LGD, REPORTS_DIR / "ifrs9_summary.md")


if __name__ == "__main__":
    main()
