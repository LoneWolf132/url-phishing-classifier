"""
config.py -- single source of truth for paths, seeds and label rules.
Every other module imports from here instead of hard-coding values.
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# This file lives in <project>/src/config.py, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# =============================================================================
# MODEL / REPORT DIRECTORIES
# =============================================================================

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_RESULTS_DIR = REPORTS_DIR / "results"

# =============================================================================
# OUTPUT FILES -- PHASE 1 (data_loader.py)
# =============================================================================

CLEANED_DATASET_PATH = DATA_PROCESSED_DIR / "cleaned_urls.csv"
CONFLICTING_URLS_PATH = DATA_PROCESSED_DIR / "conflicting_urls.csv"
INSPECTION_REPORT_PATH = DATA_PROCESSED_DIR / "inspection_report.json"

# =============================================================================
# OUTPUT FILES -- PHASE 2 (preprocessing.py)
# =============================================================================

NORMALIZED_DATASET_PATH = DATA_PROCESSED_DIR / "normalized_urls.csv"
UNPARSEABLE_URLS_PATH = DATA_PROCESSED_DIR / "unparseable_urls.csv"

# =============================================================================
# OUTPUT FILES -- PHASE 3 (feature_extraction.py)
# =============================================================================

FEATURES_DATASET_PATH = DATA_PROCESSED_DIR / "features.csv"

# =============================================================================
# OUTPUT FILES -- PHASE 5 (splitting.py)
# =============================================================================

TRAIN_SPLIT_PATH = DATA_PROCESSED_DIR / "train_split.csv"
TEST_SPLIT_PATH = DATA_PROCESSED_DIR / "test_split.csv"
SPLIT_REPORT_PATH = REPORTS_RESULTS_DIR / "split_report.json"

# =============================================================================
# OUTPUT FILES -- PHASE 7/8 (train.py)
# =============================================================================

FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
MODEL_COMPARISON_PATH = REPORTS_RESULTS_DIR / "model_comparison.json"

MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "logistic_regression.joblib",
    "decision_tree": MODELS_DIR / "decision_tree.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "xgboost": MODELS_DIR / "xgboost.joblib",
}

# The model chosen for deployment (Streamlit app) after Phase 9 comparison.
# This is set as a config value (not hard-coded in the app) so it's easy to
# change if a different model turns out to perform best on your data.
DEPLOYMENT_MODEL_NAME = "xgboost"
DEPLOYMENT_MODEL_PATH = MODELS_DIR / "deployment_model.joblib"
# =============================================================================
# OUTPUT FILES -- PROBABILITY CALIBRATION (extension of Phase 9, src/calibrate.py)
# =============================================================================

# One calibrated version of each base model, saved alongside the originals.
CALIBRATED_MODEL_PATHS = {
    name: MODELS_DIR / f"{name}_calibrated.joblib"
    for name in MODEL_PATHS
}
CALIBRATION_CURVE_DATA_PATH = REPORTS_RESULTS_DIR / "calibration_curve_data.json"

# If True, predict.py / the Streamlit app use the CALIBRATED version of
# DEPLOYMENT_MODEL_NAME when it exists (falls back to the uncalibrated model
# with a printed/displayed note if calibration hasn't been run yet).
USE_CALIBRATED_MODEL = True
# =============================================================================
# OUTPUT FILES -- PHASE 9 (evaluate.py)
# =============================================================================

EVALUATION_RESULTS_PATH = REPORTS_RESULTS_DIR / "evaluation_results.json"

# =============================================================================
# OUTPUT FILES -- PHASE 14 (error_analysis.py)
# =============================================================================

FALSE_POSITIVES_PATH = REPORTS_RESULTS_DIR / "false_positives.csv"
FALSE_NEGATIVES_PATH = REPORTS_RESULTS_DIR / "false_negatives.csv"

# =============================================================================
# URL / LABEL COLUMN CONFIGURATION
# =============================================================================

URL_COLUMN_OVERRIDE = None
LABEL_COLUMN_OVERRIDE = None

URL_COLUMN_CANDIDATES = (
    "url",
    "urls",
    "link",
    "website",
    "websites",
    "domain",
)

LABEL_COLUMN_CANDIDATES = (
    "label",
    "labels",
    "class",
    "type",
    "status",
    "target",
)

# =============================================================================
# LABEL MAPPING
# =============================================================================
# Convention used in the WHOLE project: 1 = phishing (positive class), 0 = benign.

LABEL_TEXT_MAP = {
    "benign": 0,
    "legitimate": 0,
    "legitimate url": 0,
    "good": 0,
    "safe": 0,

    "phishing": 1,
    "malicious": 1,
    "malicious url": 1,
    "bad": 1,
}

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================

TEST_SIZE = 0.2  # 20% of registered domains go to the test set

# =============================================================================
# SUSPICIOUS KEYWORDS (Phase 3 heuristic feature)
# =============================================================================

SUSPICIOUS_KEYWORDS = (
    "login", "verify", "verification", "secure", "account",
    "update", "signin", "banking", "password", "confirm",
)

# =============================================================================
# DIRECTORY CREATION
# =============================================================================


def ensure_directories() -> None:
    """Create every output folder if it doesn't exist yet (safe to call repeatedly)."""
    for folder in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_FIGURES_DIR,
        REPORTS_RESULTS_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)
