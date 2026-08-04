"""Single source of truth for paths, seeds and modelling constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

RAW_TRAIN_PATH = DATA_DIR / "application_train.csv"
LGBM_MODEL_PATH = MODELS_DIR / "lgbm_model.joblib"
TRAIN_MEDIANS_PATH = MODELS_DIR / "train_medians.joblib"
CAT_DTYPES_PATH = MODELS_DIR / "category_dtypes.joblib"

# approve/decline cutoff for the API — set to the population base default rate (8.1%, see
# reports/eda_summary.md); a real deployment would tune this against a target approval rate or
# expected loss budget, but a round, explainable number beats an unexplained one for a demo.
DECISION_THRESHOLD = 0.08

RANDOM_SEED = 42
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# 60/20/20, stratified on TARGET
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2

# IFRS 9 default assumptions — overridable per call, not hardcoded into the ECL math itself
DEFAULT_LGD = 0.45
DEFAULT_EAD_COL = "AMT_CREDIT"

# a loan is Stage 2 (lifetime ECL) if current PD has at least doubled since origination
SICR_PD_RATIO_THRESHOLD = 2.0

MLFLOW_EXPERIMENT_NAME = "credit-risk-ifrs9-engine"
