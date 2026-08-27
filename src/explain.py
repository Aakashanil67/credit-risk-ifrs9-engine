"""SHAP explainability: global feature importance and per-applicant reason codes.

SHAP (SHapley Additive exPlanations) assigns each feature a contribution to one prediction —
positive means it pushed the predicted default probability up, negative means it pushed it down —
such that the contributions sum exactly to (prediction - average prediction). That additivity is
what makes "top 3 SHAP drivers" a defensible sentence rather than a hand-wave: those three
features really did account for most of the gap between this applicant's score and the average.
"""

import argparse

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.artifacts import load_artifact_bundle
from src.config import FIGURES_DIR, RANDOM_SEED, TARGET_COL, model_bundle_dir
from src.data_loader import load_application_data
from src.features import build_lgbm_features
from src.model_profiles import ModelProfile
from src.preprocessing import split_data
from src.reason_codes import reason_codes

SHAP_SAMPLE_SIZE = 3000  # full test set (61k rows) isn't needed for a stable importance ranking


def load_model_bundle(profile: ModelProfile):
    """Load the versioned model bundle selected for an explainability run."""
    return load_artifact_bundle(model_bundle_dir(profile.value))


def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def shap_raw_scores(base_values: np.ndarray, shap_values: np.ndarray) -> np.ndarray:
    """Reconstruct LightGBM's raw margin from Tree SHAP's additive components."""
    return np.asarray(base_values) + np.asarray(shap_values).sum(axis=1)


def explanation_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """Return the untouched test fold for published explanation and calibration artefacts.

    The saved model is refit on train plus validation rows after tuning. Reusing validation here
    would therefore make the visual diagnostics in-sample, even though the headline test metrics
    remain honest.
    """
    _train, _validation, test = split_data(df, seed=RANDOM_SEED)
    return test


def validate_shap_additivity(model, X: pd.DataFrame, explanation: shap.Explanation) -> None:
    """Fail if the explanation stops reconstructing the model's raw output."""
    expected = model.predict(X, raw_score=True)
    actual = shap_raw_scores(explanation.base_values, explanation.values)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def save_current_shap_plot(out_path) -> None:
    """Call right after a shap.plots.* call — each one draws on the current pyplot figure, so
    starting a fresh figure before every plot (not just once) keeps them from overlaying."""
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SHAP plots for a versioned model bundle")
    parser.add_argument(
        "--profile",
        choices=[ModelProfile.PUBLIC_DEMO.value],
        default=ModelProfile.PUBLIC_DEMO.value,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = ModelProfile(args.profile)
    df = load_application_data()
    test = explanation_holdout(df)

    bundle = load_model_bundle(profile)
    model = bundle.model

    test_X = build_lgbm_features(test, profile=profile)
    sample = test_X.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_SEED)
    explanation = compute_shap_values(model, sample)
    validate_shap_additivity(model, sample, explanation)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.plots.beeswarm(explanation, show=False, max_display=15)
    save_current_shap_plot(FIGURES_DIR / f"{profile.value}_shap_beeswarm.png")

    plt.figure()
    shap.plots.bar(explanation, show=False, max_display=15)
    save_current_shap_plot(FIGURES_DIR / f"{profile.value}_shap_bar.png")

    print(f"wrote {profile.value}_shap_beeswarm.png and {profile.value}_shap_bar.png")

    # per-applicant waterfall + reason codes for the two highest-risk applicants in the sample
    proba = model.predict_proba(sample)[:, 1]
    riskiest = np.argsort(proba)[-2:][::-1]
    for rank, row_idx in enumerate(riskiest, start=1):
        applicant_id = test.loc[sample.index[row_idx], "SK_ID_CURR"]
        shap_row = pd.Series(explanation.values[row_idx], index=sample.columns)
        feature_row = sample.iloc[row_idx]

        plt.figure()
        shap.plots.waterfall(explanation[row_idx], show=False, max_display=10)
        save_current_shap_plot(FIGURES_DIR / f"{profile.value}_shap_waterfall_applicant_{rank}.png")

        codes = reason_codes(shap_row, feature_row)
        print(f"applicant SK_ID_CURR={applicant_id} (PD={proba[row_idx]:.3f}):")
        for code in codes:
            print(f"  - {code}")

    # Calibration uses the untouched test set, not just the SHAP sample, for stable bin estimates.
    full_test_pred = model.predict_proba(test_X)[:, 1]
    calibration = calibration_summary(test[TARGET_COL], full_test_pred)
    plot_calibration_curve(calibration, FIGURES_DIR / f"{profile.value}_calibration_curve.png")
    print(f"Brier score: {calibration['brier_score']:.4f}")


if __name__ == "__main__":
    main()
