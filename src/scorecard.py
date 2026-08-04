"""Points-based credit scorecard via optbinning — the format a credit committee actually reads.

WoE (weight of evidence) bins a continuous or categorical feature and replaces each bin with
ln(% of good outcomes in the bin / % of bad outcomes in the bin). A bin where defaults are
over-represented gets a negative WoE; a safe bin gets positive WoE. It turns any feature into a
monotonic, interpretable log-odds contribution regardless of its original scale or shape.

IV (information value) sums each bin's WoE weighted by the good/bad split it came from — one
number per feature summarising how much separating power it has. Rule of thumb used in the
industry: IV < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, > 0.5 suspiciously
strong (often a leak).

Binning is fit one variable at a time with optbinning's `OptimalBinning(solver="mip")` — the
higher-level `BinningProcess`/`Scorecard` orchestrator segfaults on this machine's ortools build
(its default CP-SAT solver path hits a broken `LinearExpr.__radd__` overload — see CLAUDE.md).
Fitting each variable directly and doing the WoE regression + PDO points math by hand sidesteps
that bug entirely and is the same computation `Scorecard` would have done internally.

Points come from PDO scaling (points-to-double-odds): a chosen baseline score corresponds to a
chosen good:bad odds, and every `pdo` points added doubles the odds of being good.
    factor = pdo / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset - factor * (intercept + sum(beta_i * WoE_i(x)))
under the convention that the logistic model predicts P(default); the minus sign flips it so
higher scores mean lower risk, matching how every bank-issued scorecard reads.
"""

from optbinning import OptimalBinning  # noqa: I001 — must import before numpy/pandas, see note below

# optbinning pulls in ortools, which bundles its own native DLLs (abseil, protobuf, highs). On
# this machine, importing numpy/pandas first and ortools second corrupts something in that DLL
# set and segfaults deep inside the MIP solver — no Python-level traceback, just exit code 139.
# Importing optbinning before numpy/pandas avoids it. Reorder at your peril; verified empirically.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_SEED, REPORTS_DIR, TARGET_COL
from src.data_loader import load_application_data
from src.preprocessing import split_data

NUMERIC_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "age_years",
    "years_employed",
    "REGION_POPULATION_RELATIVE",
    "CNT_CHILDREN",
]
CATEGORICAL_FEATURES = ["CODE_GENDER", "NAME_EDUCATION_TYPE", "FLAG_OWN_CAR"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PDO = 20
BASE_ODDS = 50
BASE_SCORE = 600


def engineer_scorecard_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "REGION_POPULATION_RELATIVE",
            "CNT_CHILDREN",
            "CODE_GENDER",
            "NAME_EDUCATION_TYPE",
            "FLAG_OWN_CAR",
        ]
    ].copy()
    out["age_years"] = -df["DAYS_BIRTH"] / 365.25
    out["years_employed"] = df["DAYS_EMPLOYED"].replace(365243, np.nan) / -365.25
    return out[FEATURES]


def fit_binnings(X: pd.DataFrame, y: pd.Series) -> dict[str, OptimalBinning]:
    binnings = {}
    for feat in FEATURES:
        dtype = "categorical" if feat in CATEGORICAL_FEATURES else "numerical"
        ob = OptimalBinning(name=feat, dtype=dtype, solver="mip")
        ob.fit(X[feat].to_numpy(), y.to_numpy())
        ob.binning_table.build()  # populates .iv / .build() output used below
        binnings[feat] = ob
    return binnings


def woe_transform(X: pd.DataFrame, binnings: dict[str, OptimalBinning]) -> pd.DataFrame:
    return pd.DataFrame(
        {feat: binnings[feat].transform(X[feat].to_numpy(), metric="woe") for feat in FEATURES},
        index=X.index,
    )


def iv_table(binnings: dict[str, OptimalBinning]) -> pd.DataFrame:
    rows = [{"name": feat, "iv": ob.binning_table.iv} for feat, ob in binnings.items()]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def points_table(binnings: dict[str, OptimalBinning], model: LogisticRegression) -> pd.DataFrame:
    factor = PDO / np.log(2)
    coefs = dict(zip(FEATURES, model.coef_[0], strict=True))
    rows = []
    for feat, ob in binnings.items():
        table = ob.binning_table.build()
        for idx, bin_row in table.iterrows():
            if idx == "Totals":  # summary row, not a real bin — "Totals" is the index, not Bin
                continue
            points = -factor * coefs[feat] * bin_row["WoE"]
            bin_label = bin_row["Bin"]
            if isinstance(bin_label, np.ndarray):
                bin_label = ", ".join(str(v) for v in bin_label)
            rows.append(
                {
                    "feature": feat,
                    "bin": bin_label,
                    "woe": bin_row["WoE"],
                    "points": round(points, 1),
                }
            )
    return pd.DataFrame(rows)


def base_points(model: LogisticRegression) -> float:
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return offset - factor * model.intercept_[0]


def score(
    X: pd.DataFrame, binnings: dict[str, OptimalBinning], model: LogisticRegression
) -> np.ndarray:
    woe = woe_transform(X, binnings)
    factor = PDO / np.log(2)
    contributions = -factor * (woe * model.coef_[0]).sum(axis=1)
    return base_points(model) + contributions


def write_scorecard_report(iv: pd.DataFrame, points: pd.DataFrame, base: float, out_path) -> None:
    lines = [
        "# Credit scorecard",
        "",
        f"WoE/IV binning via optbinning, scaled to points with PDO (points-to-double-odds): "
        f"{BASE_SCORE} points at {BASE_ODDS}:1 good:bad odds, every {PDO} points doubles the "
        f"odds of being good. Base points (from the model intercept alone): {base:.1f}.",
        "",
        "## Information value by feature",
        "",
        "IV < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, above 0.5 suspiciously "
        "strong (usually a leak worth double-checking).",
        "",
        "| feature | IV |",
        "|---|---|",
    ]
    for _, row in iv.iterrows():
        lines.append(f"| {row['name']} | {row['iv']:.4f} |")

    lines += ["", "## Points table", "", "| feature | bin | WoE | points |", "|---|---|---|---|"]
    for _, row in points.iterrows():
        lines.append(
            f"| {row['feature']} | {row['bin']} | {row['woe']:.4f} | {row['points']:+.1f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_application_data()
    train, val, _test = split_data(df, seed=RANDOM_SEED)

    train_X = engineer_scorecard_features(train)
    train_y = train[TARGET_COL]

    binnings = fit_binnings(train_X, train_y)
    iv = iv_table(binnings)
    print(iv.to_string(index=False))

    train_woe = woe_transform(train_X, binnings)
    model = LogisticRegression(random_state=RANDOM_SEED)
    model.fit(train_woe, train_y)

    points = points_table(binnings, model)
    base = base_points(model)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_scorecard_report(iv, points, base, REPORTS_DIR / "scorecard.md")

    val_X = engineer_scorecard_features(val)
    val_scores = score(val_X, binnings, model)
    print(
        f"validation score range: {val_scores.min():.0f}-{val_scores.max():.0f}, mean {val_scores.mean():.1f}"
    )


if __name__ == "__main__":
    main()
