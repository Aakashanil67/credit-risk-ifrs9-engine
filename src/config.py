"""Single source of truth for paths, seeds and modelling constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_TRAIN_PATH = DATA_DIR / "application_train.csv"

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
