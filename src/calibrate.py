"""
calibrate.py -- PHASE 9 (extension): PROBABILITY CALIBRATION

Run from the project root with:
    python -m src.calibrate

WHY: class_weight="balanced" (and XGBoost's scale_pos_weight) fixes the
decision boundary for class imbalance, but it systematically distorts the
raw predicted probabilities -- confirmed by the calibration curve, where all
four models sit below the diagonal (predicted risk is inflated relative to
true frequency) across most of the probability range.

WHAT THIS DOES: for each already-trained model, fits a small calibration
layer on top (Platt scaling / sigmoid) using a held-out calibration split
carved out of the ORIGINAL TRAINING DOMAINS (never the test domains, so this
does not reintroduce leakage). The underlying model's decision boundary and
learned patterns are NOT retrained -- only the probability mapping is
adjusted. Saves calibrated models and a before/after calibration curve.
"""

import json
from src import config
from src.splitting import build_group_column
from src.train import get_feature_columns

# Paths now live in src/config.py (single source of truth) --
# see config.CALIBRATED_MODEL_PATHS and config.CALIBRATION_CURVE_DATA_PATH.
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GroupShuffleSplit

from src import config
from src.splitting import build_group_column
from src.train import get_feature_columns

CALIBRATED_MODEL_PATHS = {
    name: config.MODELS_DIR / f"{name}_calibrated.joblib"
    for name in config.MODEL_PATHS
}
CALIBRATION_CURVE_DATA_PATH = config.REPORTS_RESULTS_DIR / "calibration_curve_data.json"


def make_calibration_split(train_df: pd.DataFrame, feature_columns: list):
    """
    Carve a calibration subset out of the TRAINING domains only, using the
    same grouped-split logic as Phase 5, so calibration fitting never sees
    the real test-set domains.
    """
    groups = build_group_column(train_df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_SEED)
    fit_idx, calib_idx = next(splitter.split(train_df, groups=groups))

    fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
    calib_df = train_df.iloc[calib_idx].reset_index(drop=True)

    fit_groups = set(groups.iloc[fit_idx])
    calib_groups = set(groups.iloc[calib_idx])
    assert len(fit_groups & calib_groups) == 0, (
        "Calibration split leaked domains between fit and calibration subsets."
    )

    X_calib = calib_df[feature_columns].values
    y_calib = calib_df["label"].values
    return X_calib, y_calib


def calibrate_model(model, X_calib, y_calib):
    """
    Wraps an ALREADY-TRAINED model with a sigmoid (Platt scaling) calibration
    layer, fit only on X_calib/y_calib. The base model itself is frozen and
    not retrained.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_calib, y_calib)
    return calibrated


def get_calibration_curve_points(y_true, y_proba, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    return mean_pred.tolist(), frac_pos.tolist()


def plot_calibration_comparison(before: dict, after: dict, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfectly calibrated")
    ax.plot(before["mean_pred"], before["frac_pos"], marker="o", label="Before calibration")
    ax.plot(after["mean_pred"], after["frac_pos"], marker="o", label="After calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve Before/After: {model_name}")
    ax.legend()
    fig.tight_layout()
    path = config.REPORTS_FIGURES_DIR / f"calibration_before_after_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    config.ensure_directories()

    train_df = pd.read_csv(config.TRAIN_SPLIT_PATH)
    test_df = pd.read_csv(config.TEST_SPLIT_PATH)
    with open(config.FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    X_calib, y_calib = make_calibration_split(train_df, feature_columns)
    X_test = test_df[feature_columns].values
    y_test = test_df["label"].values

    print(f"[calibrate] Calibration set size: {len(y_calib):,}")
    print(f"[calibrate] Test set size:        {len(y_test):,}")

    all_curve_data = {}

    for name, path in config.MODEL_PATHS.items():
        if not path.exists():
            print(f"[skip] {name}: model file not found at {path}")
            continue

        print(f"\n=== Calibrating: {name} ===")
        model = joblib.load(path)

        y_proba_before = model.predict_proba(X_test)[:, 1]
        mean_pred_before, frac_pos_before = get_calibration_curve_points(y_test, y_proba_before)

        calibrated_model = calibrate_model(model, X_calib, y_calib)
        y_proba_after = calibrated_model.predict_proba(X_test)[:, 1]
        mean_pred_after, frac_pos_after = get_calibration_curve_points(y_test, y_proba_after)

        print(f"   Mean predicted probability BEFORE: {np.mean(y_proba_before):.4f}")
        print(f"   Mean predicted probability AFTER:  {np.mean(y_proba_after):.4f}")
        print(f"   Actual positive rate (test set):   {y_test.mean():.4f}")

        before = {"mean_pred": mean_pred_before, "frac_pos": frac_pos_before}
        after = {"mean_pred": mean_pred_after, "frac_pos": frac_pos_after}
        all_curve_data[name] = {"before": before, "after": after}

        fig_path = plot_calibration_comparison(before, after, name)
        print(f"   [saved figure] {fig_path}")

        joblib.dump(calibrated_model, CALIBRATED_MODEL_PATHS[name])
        print(f"   [saved] {CALIBRATED_MODEL_PATHS[name]}")
        joblib.dump(calibrated_model, config.CALIBRATED_MODEL_PATHS[name])
        print(f"   [saved] {config.CALIBRATED_MODEL_PATHS[name]}")

    with open(config.CALIBRATION_CURVE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_curve_data, f, indent=2)
    print(f"\n[saved] {config.CALIBRATION_CURVE_DATA_PATH}")

    with open(CALIBRATION_CURVE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_curve_data, f, indent=2)
    print(f"\n[saved] {CALIBRATION_CURVE_DATA_PATH}")

    print("\n[done] Calibration finished.")
    


if __name__ == "__main__":
    main()