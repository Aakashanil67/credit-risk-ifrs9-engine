"""Logistic regression baseline — the interpretable benchmark every fancier model has to beat.

15 features chosen for interpretability, not maximum signal: a credit committee needs to be able
to read every coefficient as an odds ratio and explain it. LightGBM (src/train_lgbm.py) gets the
full feature set later.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import FIGURES_DIR, RANDOM_SEED, REPORTS_DIR, TARGET_COL
from src.data_loader import load_application_data
from src.preprocessing import split_data

FEATURES = [
    "log_income",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "age_years",
    "years_employed",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "REGION_POPULATION_RELATIVE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "is_male",
    "owns_car",
    "owns_realty",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_income"] = np.log1p(df["AMT_INCOME_TOTAL"])
    out["AMT_CREDIT"] = df["AMT_CREDIT"]
    out["AMT_ANNUITY"] = df["AMT_ANNUITY"]
    out["AMT_GOODS_PRICE"] = df["AMT_GOODS_PRICE"]
    out["age_years"] = -df["DAYS_BIRTH"] / 365.25
    # 365243 is Home Credit's sentinel for "not currently employed" — not a real tenure
    out["years_employed"] = df["DAYS_EMPLOYED"].replace(365243, np.nan) / -365.25
    out["EXT_SOURCE_1"] = df["EXT_SOURCE_1"]
    out["EXT_SOURCE_2"] = df["EXT_SOURCE_2"]
    out["EXT_SOURCE_3"] = df["EXT_SOURCE_3"]
    out["REGION_POPULATION_RELATIVE"] = df["REGION_POPULATION_RELATIVE"]
    out["CNT_CHILDREN"] = df["CNT_CHILDREN"]
    out["CNT_FAM_MEMBERS"] = df["CNT_FAM_MEMBERS"]
    out["is_male"] = (df["CODE_GENDER"] == "M").astype(int)
    out["owns_car"] = (df["FLAG_OWN_CAR"] == "Y").astype(int)
    out["owns_realty"] = (df["FLAG_OWN_REALTY"] == "Y").astype(int)
    return out


def fill_with_train_median(train_X: pd.DataFrame, val_X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = train_X.median()
    return train_X.fillna(medians), val_X.fillna(medians)


def fit_logit(train_X: pd.DataFrame, train_y: pd.Series) -> sm.Logit:
    design = sm.add_constant(train_X)
    model = sm.Logit(train_y, design).fit(disp=False)
    return model


def coefficient_table(model: sm.Logit) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "coef": model.params,
            "odds_ratio": np.exp(model.params),
            "p_value": model.pvalues,
        }
    )
    return table.drop("const").sort_values("p_value")


def plot_roc(y_true: pd.Series, y_pred: np.ndarray, auc: float, out_path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"logistic baseline (AUC={auc:.4f})", color="#2a6f97")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC — logistic regression baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_interpretation(table: pd.DataFrame, auc: float, out_path) -> None:
    significant = table[table["p_value"] < 0.05].reindex(
        table[table["p_value"] < 0.05]["coef"].abs().sort_values(ascending=False).index
    )
    top5 = significant.head(5)

    lines = [
        "# Logistic regression baseline",
        "",
        f"Validation AUC: **{auc:.4f}**. Trained on {len(FEATURES)} hand-picked, interpretable "
        "features — not the full 122-column table LightGBM gets later.",
        "",
        "## Reading the coefficients",
        "",
        "Each odds ratio is how much an applicant's odds of default multiply by for a one-unit "
        "increase in that feature, holding everything else fixed. An odds ratio above 1 raises "
        "default risk; below 1 lowers it.",
        "",
        "Caveat: features aren't standardised, so 'one unit' means very different things across "
        "rows — a one-unit move in `REGION_POPULATION_RELATIVE` (range ~0-0.07) spans nearly its "
        "whole distribution, while a one-unit move in `AMT_CREDIT` (rand) is negligible, which is "
        "why its odds ratio rounds to 1.0000 even though the coefficient is real and significant. "
        "Compare effect sizes within a feature's own range, not across features by raw odds ratio.",
        "",
        "## Top 5 statistically significant drivers",
        "",
    ]
    for feat, row in top5.iterrows():
        direction = "increases" if row["odds_ratio"] > 1 else "decreases"
        pct_effect = abs(row["odds_ratio"] - 1) * 100
        lines.append(
            f"- **{feat}**: odds ratio {row['odds_ratio']:.3f} (p={row['p_value']:.2e}). "
            f"Each one-unit increase {direction} the odds of default by {pct_effect:.1f}%."
        )
    lines += [
        "",
        "## Full coefficient table",
        "",
        "| feature | coef | odds ratio | p-value |",
        "|---|---|---|---|",
    ]
    for feat, row in table.iterrows():
        lines.append(f"| {feat} | {row['coef']:.4f} | {row['odds_ratio']:.4f} | {row['p_value']:.2e} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_application_data()
    train, val, _test = split_data(df, seed=RANDOM_SEED)

    train_X, val_X = fill_with_train_median(engineer_features(train), engineer_features(val))
    train_y, val_y = train[TARGET_COL], val[TARGET_COL]

    model = fit_logit(train_X, train_y)
    val_pred = model.predict(sm.add_constant(val_X, has_constant="add"))
    auc = roc_auc_score(val_y, val_pred)
    print(f"validation AUC: {auc:.4f}")

    table = coefficient_table(model)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc(val_y, val_pred, auc, FIGURES_DIR / "baseline_roc.png")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_interpretation(table, auc, REPORTS_DIR / "baseline_interpretation.md")


if __name__ == "__main__":
    main()
