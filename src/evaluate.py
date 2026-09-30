"""
evaluate.py -- PHASE 9: MODEL EVALUATION

Run from the project root with:
    python -m src.evaluate

Loads all four trained models (Phase 7/8) and the held-out test set
(Phase 5 -- a set of domains the models have NEVER seen). Computes the full
metric set required by Phase 6, saves figures to reports/figures/ and
numerical results to reports/results/evaluation_results.json.

CRITICAL RULE: everything here is computed on test_split.csv only. Training
data is never touched in this file. Evaluating on training data would report
memorisation, not generalisation.
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI needed; we only save figures to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve,
)

from src import config
from src.train import get_feature_columns


def load_test_set():
    test_df = pd.read_csv(config.TEST_SPLIT_PATH)
    with open(config.FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    X_test = test_df[feature_columns].values
    y_test = test_df["label"].values
    return test_df, X_test, y_test, feature_columns


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """
    All metrics required by Phase 6. Accuracy alone is misleading on an
    imbalanced dataset (always predicting "benign" already scores ~77%
    accuracy here while catching zero phishing), so we report the full set.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "matthews_corrcoef": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "false_positive_rate": round(float(false_positive_rate), 4),
        "false_negative_rate": round(float(false_negative_rate), 4),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Phishing"])
    ax.set_yticklabels(["Benign", "Phishing"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    path = config.REPORTS_FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc_curve(results: dict, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random guess")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = config.REPORTS_FIGURES_DIR / "roc_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_pr_curve(results: dict, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        precision, recall, _ = precision_recall_curve(y_test, r["y_proba"])
        ax.plot(recall, precision, label=f"{name} (PR-AUC={r['metrics']['pr_auc']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves"); ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path = config.REPORTS_FIGURES_DIR / "precision_recall_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

def plot_calibration_curve(results: dict, y_test, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfectly calibrated")
    
    for name, r in results.items():
        prob_true, prob_pred = calibration_curve(y_test, r["y_proba"], n_bins=n_bins, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", markersize=4, label=name)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves (Reliability Diagram)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = config.REPORTS_FIGURES_DIR / "calibration_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

def plot_feature_importance(model, feature_columns, model_name: str):
    """Only meaningful for tree-based models; skipped for Logistic Regression here
    (its equivalent is coefficient magnitude, which is not directly comparable)."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:15]  # top 15 for readability
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(
        [feature_columns[i] for i in order][::-1],
        importances[order][::-1],
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 15 Feature Importances: {model_name}")
    fig.tight_layout()
    path = config.REPORTS_FIGURES_DIR / f"feature_importance_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    config.ensure_directories()
    test_df, X_test, y_test, feature_columns = load_test_set()
    print(f"[load] Test set: {len(test_df):,} rows, {len(feature_columns)} features")

    results = {}
    for name, path in config.MODEL_PATHS.items():
        if not path.exists():
            print(f"[skip] {name}: model file not found at {path} (run src.train first)")
            continue

        print(f"\n=== Evaluating: {name} ===")
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        for k, v in metrics.items():
            print(f"   {k}: {v}")

        cm_path = plot_confusion_matrix(y_test, y_pred, name)
        fi_path = plot_feature_importance(model, feature_columns, name)

        results[name] = {"metrics": metrics, "y_proba": y_proba}
        print(f"   [saved figure] {cm_path}")
        if fi_path:
            print(f"   [saved figure] {fi_path}")

    if not results:
        print("[error] No trained models found. Run 'python -m src.train' first.")
        return

    roc_path = plot_roc_curve(results, y_test)
    pr_path = plot_pr_curve(results, y_test)
    cal_path = plot_calibration_curve(results, y_test)
    print(f"\n[saved figure] {roc_path}")
    print(f"[saved figure] {pr_path}")
    print(f"[saved figure] {cal_path}")

    # Model comparison table -- printed, saved, and NOT auto-declaring a
    # "winner" beyond reporting the numbers (per project rule: no unsupported
    # complexity-equals-best claims).
    print("\n=== MODEL COMPARISON TABLE ===")
    comparison_rows = []
    for name, r in results.items():
        row = {"model": name, **r["metrics"]}
        row.pop("confusion_matrix", None)
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows).set_index("model")
    print(comparison_df.to_string())

    output = {name: r["metrics"] for name, r in results.items()}
    with open(config.EVALUATION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[saved] {config.EVALUATION_RESULTS_PATH}")
    
    best_by_pr_auc = comparison_df["pr_auc"].idxmax()
    print(f"\n[note] Highest PR-AUC on this run: '{best_by_pr_auc}'. "
          f"PR-AUC is used (rather than accuracy) because it is more informative "
          f"under class imbalance. This is a data point, not an automatic "
          f"verdict -- inspect the full table above before deciding the "
          f"deployment model (config.DEPLOYMENT_MODEL_NAME).")

    print("\n[done] Phase 9 finished.")


if __name__ == "__main__":
    main()
