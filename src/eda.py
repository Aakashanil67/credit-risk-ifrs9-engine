"""Exploratory analysis: target imbalance, missingness, key feature distributions, target correlations."""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR, TARGET_COL
from src.data_loader import load_application_data, missing_value_report, target_balance

KEY_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "REGION_POPULATION_RELATIVE",
]


def plot_target_imbalance(df: pd.DataFrame, out_path) -> None:
    balance = target_balance(df)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(balance.index, balance.values, color=["#2a6f97", "#c1121f"])
    ax.set_ylabel("share of applicants")
    ax.set_title(f"Target imbalance ({balance['defaulted (1)']:.1%} defaulted)")
    for i, v in enumerate(balance.values):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_missingness_heatmap(missing: pd.DataFrame, out_path, top_n: int = 30) -> None:
    worst = missing.head(top_n)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.barh(worst.index[::-1], worst["missing_pct"][::-1], color="#5c677d")
    ax.set_xlabel("% missing")
    ax.set_title(f"{top_n} columns with the most missing values")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_key_feature_distributions(df: pd.DataFrame, out_path) -> None:
    # DAYS_BIRTH / DAYS_EMPLOYED are negative day-counts relative to application date —
    # flip to positive years so the histograms read naturally.
    plotted = df[KEY_FEATURES].copy()
    plotted["DAYS_BIRTH"] = -plotted["DAYS_BIRTH"] / 365.25
    plotted["DAYS_EMPLOYED"] = plotted["DAYS_EMPLOYED"].replace(365243, pd.NA) / -365.25

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, col in zip(axes.flat, KEY_FEATURES, strict=True):
        series = plotted[col].dropna()
        clip = series.quantile(0.99)  # a handful of extreme incomes blow out the axis otherwise
        ax.hist(series.clip(upper=clip), bins=40, color="#2a6f97")
        ax.set_title(col, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def target_correlations(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes("number")
    corr = numeric.corr()[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
    return corr


def write_eda_summary(df: pd.DataFrame, missing: pd.DataFrame, corr: pd.Series, out_path) -> None:
    balance = target_balance(df)
    top_corr = corr.head(5)
    anomaly_365243 = (df["DAYS_EMPLOYED"] == 365243).mean()

    lines = [
        "# EDA summary",
        "",
        "## Why accuracy is the wrong metric",
        "",
        f"{balance['defaulted (1)']:.1%} of applicants defaulted. A model that predicts "
        f"'repaid' for every single applicant hits {balance['repaid (0)']:.1%} accuracy while "
        "identifying zero of the defaults a lender actually cares about. AUC, precision-recall, "
        "and the KS statistic (computed later against LightGBM) all separate the classes; "
        "accuracy collapses them.",
        "",
        "## Five findings",
        "",
        f"1. **Missingness is structural, not random.** {len(missing)} of {df.shape[1]} columns "
        f"have missing values, and the worst ({missing.index[0]}) is missing "
        f"{missing.iloc[0]['missing_pct']}% of the time. These are almost all building/apartment "
        "features that only exist for applicants who reported living in an apartment complex — "
        "missing is informative, not a data-entry gap, which argues against dropping the columns.",
        "",
        f"2. **`DAYS_EMPLOYED` has a sentinel-value bug.** {anomaly_365243:.1%} of rows carry the "
        "value 365243 (1,000 years), which is Home Credit's placeholder for "
        "'not currently employed', not a real employment tenure. Left untreated it wrecks any "
        "model that uses employment length linearly.",
        "",
        f"3. **The three `EXT_SOURCE_*` columns dominate the correlation table.** "
        f"{top_corr.index[0]} correlates {top_corr.iloc[0]:.3f} with `TARGET`, more than any "
        "engineered feature in this table — they're external credit-bureau scores, and the model "
        "will lean on them heavily.",
        "",
        "4. **Income and credit amount are heavily right-skewed.** A handful of applicants report "
        "incomes in the tens of millions; the histograms clip at the 99th percentile so the bulk "
        "of the distribution is visible at all. Tree models handle this natively, but the logistic "
        "baseline will need a log transform.",
        "",
        f"5. **`DAYS_BIRTH` correlates {corr.get('DAYS_BIRTH', 0):.3f} with `TARGET`** — younger "
        "applicants default more often, consistent with less credit history to underwrite "
        "against, not a causal age effect the model should be read as endorsing.",
        "",
        "## Top correlations with TARGET",
        "",
        "| feature | correlation |",
        "|---|---|",
    ]
    for feat, val in top_corr.items():
        lines.append(f"| {feat} | {val:.3f} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_application_data()
    missing = missing_value_report(df)
    corr = target_correlations(df)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_target_imbalance(df, FIGURES_DIR / "target_imbalance.png")
    plot_missingness_heatmap(missing, FIGURES_DIR / "missingness.png")
    plot_key_feature_distributions(df, FIGURES_DIR / "key_feature_distributions.png")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_eda_summary(df, missing, corr, REPORTS_DIR / "eda_summary.md")
    print("wrote 3 figures to reports/figures/ and reports/eda_summary.md")


if __name__ == "__main__":
    main()
