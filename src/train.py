"""
train.py -- PHASES 6, 7, 8: CLASS IMBALANCE, MODELS, HYPERPARAMETER TUNING

Run from the project root with:
    python -m src.train

WHAT THIS DOES:
  1. Loads train_split.csv / test_split.csv (Phase 5 output).
  2. Selects the numeric feature columns (drops text/id columns).
  3. Trains four models in increasing complexity:
       - Logistic Regression  (linear baseline)
       - Decision Tree        (simple nonlinear baseline)
       - Random Forest        (main classical ensemble)
       - XGBoost              (gradient-boosted trees)
  4. Uses class_weight="balanced" (or XGBoost's scale_pos_weight equivalent)
     instead of SMOTE, per project rules -- see the note below.
  5. Runs a SMALL, documented hyperparameter search for Random Forest and
     XGBoost using GroupKFold cross-validation (so leakage prevention holds
     during tuning too, not just at the final train/test split).
  6. Saves every trained model + the exact feature column order to disk.

WHY NOT SMOTE (Phase 6 requirement):
SMOTE synthesizes new "fake" minority-class examples by interpolating between
real ones. For URL features, an interpolated feature vector does not
correspond to any real, parseable URL -- it's a point in feature space with
no clear real-world meaning, and it can quietly leak information across the
train/test boundary when combined with grouped splitting. class_weight is
simpler, has no such risk, and lets the model learn from only real examples,
just penalising mistakes on the minority (phishing) class more heavily.

WHY NOT JUST ACCURACY:
This dataset is imbalanced (roughly 77%/23% in our Phase 1 run). A model
that always predicts "benign" would already score ~77% accuracy while being
useless -- it would catch zero phishing URLs. That's why Phase 9 evaluates
with precision, recall, F1, ROC-AUC, PR-AUC and MCC instead of relying on
accuracy alone.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
import xgboost as xgb

from src import config
from src.splitting import build_group_column

# Columns that are identifiers or text, never fed to a model as-is.
NON_FEATURE_COLUMNS = {
    "original_url", "label", "registered_domain", "subdomain", "suffix",
}


def get_feature_columns(df: pd.DataFrame) -> list:
    """All numeric feature columns, in a fixed, saved order."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def load_splits():
    train_df = pd.read_csv(config.TRAIN_SPLIT_PATH)
    test_df = pd.read_csv(config.TEST_SPLIT_PATH)
    return train_df, test_df


# ----------------------------------------------------------------------------
# MODEL 1: Logistic Regression -- linear baseline
# ----------------------------------------------------------------------------
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


# ----------------------------------------------------------------------------
# MODEL 2: Decision Tree -- simple nonlinear baseline
# ----------------------------------------------------------------------------
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        max_depth=15,  # a modest cap: an unbounded tree overfits badly here
    )
    model.fit(X_train, y_train)
    return model


# ----------------------------------------------------------------------------
# MODEL 3: Random Forest -- main classical ensemble, with small tuned grid
# ----------------------------------------------------------------------------
def train_random_forest(X_train, y_train, groups_train, quick: bool = False):
    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )

    # A SMALL, understandable grid (per Phase 8 instructions) -- not an
    # exhaustive search. Each parameter controls a specific, explainable
    # trade-off:
    #   n_estimators    -> more trees = more stable but slower
    #   max_depth       -> deeper trees fit more detail but risk overfitting
    #   min_samples_leaf-> larger leaves = smoother, less overfit boundaries
    #   max_features    -> fewer features per split = more tree diversity
    param_grid = {
        "n_estimators": [200] if quick else [200, 400],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt"],
    }

    # GroupKFold ensures tuning itself never leaks a domain across its own
    # internal folds -- the same safety principle as Phase 5, applied to
    # cross-validation.
    cv = GroupKFold(n_splits=3)
    search = GridSearchCV(
        base_model,
        param_grid,
        scoring="average_precision",  # PR-AUC: appropriate for imbalanced data
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=groups_train)
    print(f"[random_forest] Best params: {search.best_params_}")
    print(f"[random_forest] Best CV PR-AUC: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_


# ----------------------------------------------------------------------------
# MODEL 4: XGBoost -- gradient-boosted tree comparison, with small tuned grid
# ----------------------------------------------------------------------------
def train_xgboost(X_train, y_train, groups_train, quick: bool = False):
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    # scale_pos_weight is XGBoost's equivalent of class_weight="balanced":
    # it up-weights errors on the minority (phishing) class.
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.RANDOM_SEED,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )

    param_grid = {
        "max_depth": [4, 6],
        "learning_rate": [0.1] if quick else [0.05, 0.1],
        "n_estimators": [200],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    cv = GroupKFold(n_splits=3)
    search = GridSearchCV(
        base_model,
        param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=groups_train)
    print(f"[xgboost] Best params: {search.best_params_}")
    print(f"[xgboost] Best CV PR-AUC: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def main(quick: bool = False) -> None:
    """
    quick=True runs a much smaller hyperparameter grid, useful for a fast
    end-to-end smoke test of the whole pipeline before committing to the
    full run on ~400,000+ training rows.
    """
    config.ensure_directories()
    train_df, test_df = load_splits()
    print(f"[load] Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

    feature_columns = get_feature_columns(train_df)
    print(f"[features] Using {len(feature_columns)} feature columns:")
    print(f"   {feature_columns}")

    X_train = train_df[feature_columns].values
    y_train = train_df["label"].values
    groups_train = build_group_column(train_df).values

    comparison = {}
    models = {}

    print("\n=== MODEL 1: Logistic Regression ===")
    t0 = time.time()
    models["logistic_regression"] = train_logistic_regression(X_train, y_train)
    comparison["logistic_regression"] = {"train_seconds": round(time.time() - t0, 2)}

    print("\n=== MODEL 2: Decision Tree ===")
    t0 = time.time()
    models["decision_tree"] = train_decision_tree(X_train, y_train)
    comparison["decision_tree"] = {"train_seconds": round(time.time() - t0, 2)}

    print("\n=== MODEL 3: Random Forest (tuned) ===")
    t0 = time.time()
    rf_model, rf_params = train_random_forest(X_train, y_train, groups_train, quick=quick)
    models["random_forest"] = rf_model
    comparison["random_forest"] = {
        "train_seconds": round(time.time() - t0, 2),
        "best_params": rf_params,
    }

    print("\n=== MODEL 4: XGBoost (tuned) ===")
    t0 = time.time()
    xgb_model, xgb_params = train_xgboost(X_train, y_train, groups_train, quick=quick)
    models["xgboost"] = xgb_model
    comparison["xgboost"] = {
        "train_seconds": round(time.time() - t0, 2),
        "best_params": xgb_params,
    }

    # Save every model + the feature column order (Phase 12: reproducibility).
    for name, model in models.items():
        joblib.dump(model, config.MODEL_PATHS[name])
        print(f"[saved] {config.MODEL_PATHS[name]}")

    with open(config.FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)
    print(f"[saved] {config.FEATURE_COLUMNS_PATH}")

    with open(config.MODEL_COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"[saved] {config.MODEL_COMPARISON_PATH}")

    print("\n[done] Phase 6/7/8 finished. Run 'python -m src.evaluate' next "
          "to compare these four models on the held-out test set.")


if __name__ == "__main__":
    import sys
    quick_mode = "--quick" in sys.argv
    main(quick=quick_mode)
