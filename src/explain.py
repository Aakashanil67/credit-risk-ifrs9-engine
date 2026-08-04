"""SHAP explainability: global feature importance and per-applicant reason codes.

SHAP (SHapley Additive exPlanations) assigns each feature a contribution to one prediction —
positive means it pushed the predicted default probability up, negative means it pushed it down —
such that the contributions sum exactly to (prediction - average prediction). That additivity is
what makes "top 3 SHAP drivers" a defensible sentence rather than a hand-wave: those three
features really did account for most of the gap between this applicant's score and the average.
"""

import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.config import FIGURES_DIR, LGBM_MODEL_PATH, RANDOM_SEED, TARGET_COL
from src.data_loader import load_application_data
from src.features import build_lgbm_features
from src.preprocessing import split_data

SHAP_SAMPLE_SIZE = (
    3000  # full validation set (61k rows) isn't needed for a stable importance ranking
)

# Human-readable descriptions for the features that show up most often in the global ranking.
# Anything not listed here falls back to the raw column name — this covers what actually mattered,
# not a hand-authored translation of all 122 raw columns.
FEATURE_DESCRIPTIONS = {
    "NAME_CONTRACT_TYPE": "loan type (cash vs revolving)",
    "EXT_SOURCE_1": "external credit bureau score (source 1)",
    "EXT_SOURCE_2": "external credit bureau score (source 2)",
    "EXT_SOURCE_3": "external credit bureau score (source 3)",
    "AMT_CREDIT": "loan amount",
    "AMT_INCOME_TOTAL": "reported income",
    "AMT_ANNUITY": "monthly loan repayment (annuity)",
    "AMT_GOODS_PRICE": "price of the goods being financed",
    "DAYS_BIRTH": "applicant age",
    "DAYS_EMPLOYED": "length of current employment",
    "DAYS_REGISTRATION": "time since last registration change",
    "DAYS_ID_PUBLISH": "time since ID document was issued",
    "REGION_POPULATION_RELATIVE": "population density of home region",
    "REGION_RATING_CLIENT": "region risk rating",
    "REGION_RATING_CLIENT_W_CITY": "region risk rating (city-adjusted)",
    "CODE_GENDER": "gender",
    "NAME_EDUCATION_TYPE": "education level",
    "NAME_INCOME_TYPE": "income type",
    "NAME_FAMILY_STATUS": "family status",
    "OCCUPATION_TYPE": "occupation",
    "ORGANIZATION_TYPE": "employer type",
    "CNT_CHILDREN": "number of children",
    "CNT_FAM_MEMBERS": "family size",
    "OWN_CAR_AGE": "age of owned car",
    "FLAG_OWN_CAR": "car ownership",
    "FLAG_OWN_REALTY": "property ownership",
}


def humanize_feature(name: str) -> str:
    return FEATURE_DESCRIPTIONS.get(name, name.replace("_", " ").lower())


def load_or_train_model():
    if not LGBM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{LGBM_MODEL_PATH} not found — run `python -m src.train_lgbm` first to train and save it."
        )
    return joblib.load(LGBM_MODEL_PATH)


def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def save_current_shap_plot(out_path) -> None:
    """Call right after a shap.plots.* call — each one draws on the current pyplot figure, so
    starting a fresh figure before every plot (not just once) keeps them from overlaying."""
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def reason_codes(
    shap_row: pd.Series, feature_row: pd.Series, train_medians: pd.Series, top_n: int = 3
) -> list[str]:
    """Turn the top-`top_n` SHAP drivers for one applicant into plain-English sentences."""
    top_features = shap_row.abs().sort_values(ascending=False).head(top_n).index

    sentences = []
    for feat in top_features:
        shap_val = shap_row[feat]
        value = feature_row[feat]
        desc = humanize_feature(feat)

        if pd.isna(value):
            # e.g. a bureau score the applicant's file doesn't have yet — "low" would be a lie
            clause = f"missing {desc}"
        elif (
            isinstance(value, int | float | np.integer | np.floating)
            and feat in train_medians.index
        ):
            qualifier = "high" if value > train_medians[feat] else "low"
            clause = f"{qualifier} {desc}"
        else:
            clause = f"{desc} of {value}"

        verb = "raises" if shap_val > 0 else "lowers"
        sentences.append(f"{clause[0].upper()}{clause[1:]} {verb} the estimated default risk.")

    return sentences


def calibration_summary(y_true: pd.Series, y_pred: np.ndarray, n_bins: int = 10) -> dict:
    """How well predicted PDs match observed default rates — separate question from discrimination.

    AUC/Gini/KS measure whether the model *ranks* risky applicants above safe ones; calibration
    measures whether a predicted PD of, say, 20% actually corresponds to roughly 20% of those
    applicants defaulting. A model can rank perfectly (AUC 1.0) while being badly calibrated
    (e.g. every prediction off by a constant factor) — IFRS 9 ECL depends on calibration, not
    just ranking, because ECL = PD x LGD x EAD uses the raw PD value, not its rank.
    """
    observed, predicted = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, y_pred)
    return {"observed": observed, "predicted": predicted, "brier_score": brier}


def plot_calibration_curve(calibration: dict, out_path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        calibration["predicted"],
        calibration["observed"],
        marker="o",
        color="#2a6f97",
        label=f"LightGBM (Brier={calibration['brier_score']:.4f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.set_xlabel("predicted PD (bin mean)")
    ax.set_ylabel("observed default rate (bin mean)")
    ax.set_title("Calibration — predicted vs observed default rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    df = load_application_data()
    train, val, _test = split_data(df, seed=RANDOM_SEED)

    model = load_or_train_model()

    val_X = build_lgbm_features(val)
    sample = val_X.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_SEED)
    explanation = compute_shap_values(model, sample)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.plots.beeswarm(explanation, show=False, max_display=15)
    save_current_shap_plot(FIGURES_DIR / "shap_beeswarm.png")

    plt.figure()
    shap.plots.bar(explanation, show=False, max_display=15)
    save_current_shap_plot(FIGURES_DIR / "shap_bar.png")

    print("wrote reports/figures/shap_beeswarm.png and shap_bar.png")

    # per-applicant waterfall + reason codes for the two highest-risk applicants in the sample
    train_X = build_lgbm_features(train)
    train_medians = train_X.select_dtypes("number").median()

    proba = model.predict_proba(sample)[:, 1]
    riskiest = np.argsort(proba)[-2:][::-1]
    for rank, row_idx in enumerate(riskiest, start=1):
        applicant_id = val.loc[sample.index[row_idx], "SK_ID_CURR"]
        shap_row = pd.Series(explanation.values[row_idx], index=sample.columns)
        feature_row = sample.iloc[row_idx]

        plt.figure()
        shap.plots.waterfall(explanation[row_idx], show=False, max_display=10)
        save_current_shap_plot(FIGURES_DIR / f"shap_waterfall_applicant_{rank}.png")

        codes = reason_codes(shap_row, feature_row, train_medians)
        print(f"applicant SK_ID_CURR={applicant_id} (PD={proba[row_idx]:.3f}):")
        for code in codes:
            print(f"  - {code}")

    # calibration on the full validation set, not just the SHAP sample — more stable bin estimates
    full_val_pred = model.predict_proba(val_X)[:, 1]
    calibration = calibration_summary(val[TARGET_COL], full_val_pred)
    plot_calibration_curve(calibration, FIGURES_DIR / "calibration_curve.png")
    print(f"Brier score: {calibration['brier_score']:.4f}")


if __name__ == "__main__":
    main()
