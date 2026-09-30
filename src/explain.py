"""
explain.py -- PHASE 10: SHAP EXPLAINABILITY

Run from the project root with:
    python -m src.explain

Uses SHAP's TreeExplainer on the deployment model (config.DEPLOYMENT_MODEL_NAME,
a tree-based model) to explain:
  1. Global feature importance -- which features matter most across the
     whole test set.
  2. A SHAP summary plot -- how each feature's value pushes predictions
     toward "phishing" or "benign".
  3. A local explanation for a single URL -- the exact features that drove
     ONE specific prediction, which is what the Streamlit app displays.

IMPORTANT (per project rules): SHAP explains what influenced the MODEL'S
prediction. A feature having high importance does NOT mean that feature,
on its own, proves a URL is malicious -- it means the model learned a
correlation between that feature and the label in the training data.
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config


def load_deployment_model():
    model_path = config.MODEL_PATHS[config.DEPLOYMENT_MODEL_NAME]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Deployment model not found at {model_path}. "
            f"Run 'python -m src.train' first."
        )
    return joblib.load(model_path)


def load_feature_columns():
    with open(config.FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_explainer(model):
    """
    TreeExplainer works directly on tree-based models (Decision Tree, Random
    Forest, XGBoost) and computes exact SHAP values much faster than the
    general-purpose KernelExplainer. This is why the deployment model must
    be tree-based.
    """
    return shap.TreeExplainer(model)


def global_feature_importance(explainer, X_sample, feature_columns) -> pd.DataFrame:
    """
    Mean absolute SHAP value per feature across a sample of rows -- a
    model-agnostic-style ranking of which features matter most overall.
    """
    shap_values = explainer.shap_values(X_sample)
    # Some models/SHAP versions return a list for binary classification
    # ([benign_shap, phishing_shap]); we want the "phishing" (class 1) side.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance_df, shap_values


def plot_dependence(shap_values, X_sample, feature_columns, feature: str,
                     interaction_feature: str = "auto"):
    """
    A SHAP dependence plot shows how ONE feature's contribution (its SHAP
    value) changes as that feature's value changes -- and colors each point
    by a second, automatically-chosen "interaction" feature.

    WHY THIS MATTERS: a feature's overall correlation with the label
    (checked with a simple groupby, e.g. "phishing ratio when path is
    empty") is a GLOBAL, MARGINAL summary. But a tree-based model can use
    the SAME feature differently depending on what OTHER features are
    present in that row -- e.g. "path_entropy=0 pushes toward phishing,
    but only when the domain also has very few dots and no subdomain."
    A dependence plot is how you see that interaction directly, instead of
    guessing at which combination of features might explain a surprising
    local prediction.
    """
    feature_idx = feature_columns.index(feature)
    fig = plt.figure(figsize=(7, 5))
    shap.dependence_plot(
        feature_idx, shap_values, X_sample, feature_names=feature_columns,
        interaction_index=interaction_feature, show=False,
    )
    fig.tight_layout()
    path = config.REPORTS_FIGURES_DIR / f"shap_dependence_{feature}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_summary(shap_values, X_sample, feature_columns):
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_columns, show=False
    )
    fig.tight_layout()
    path = config.REPORTS_FIGURES_DIR / "shap_summary_plot.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def explain_single_prediction(explainer, model, feature_row: np.ndarray,
                               feature_columns: list, top_n: int = 5) -> dict:
    """
    Local explanation for ONE URL's feature vector. Returns a dict with the
    prediction, probability, and the top contributing features -- exactly
    the shape the Streamlit app (Phase 11) needs to display.

    feature_row: a 1D array in the SAME order as feature_columns.
    """
    X_single = feature_row.reshape(1, -1)
    proba = float(model.predict_proba(X_single)[0, 1])
    prediction = "PHISHING" if proba >= 0.5 else "BENIGN"

    shap_values = explainer.shap_values(X_single)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    row_shap = shap_values[0]

    contributions = pd.DataFrame({
        "feature": feature_columns,
        "value": feature_row,
        "shap_value": row_shap,
    })
    # Sort by absolute impact -- the features that pushed the prediction the
    # most in EITHER direction (toward phishing or toward benign).
    contributions["abs_shap"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_shap", ascending=False)
    top_contributions = contributions.head(top_n).drop(columns="abs_shap")

    return {
        "prediction": prediction,
        "phishing_probability": round(proba, 4),
        "top_contributing_features": top_contributions.to_dict(orient="records"),
    }


def main() -> None:
    config.ensure_directories()
    model = load_deployment_model()
    feature_columns = load_feature_columns()

    test_df = pd.read_csv(config.TEST_SPLIT_PATH)
    X_test = test_df[feature_columns].values

    # Use a sample for global explanations -- SHAP on the full test set can
    # be slow; a random sample of a few thousand rows gives a stable picture.
    sample_size = min(2000, len(X_test))
    rng = np.random.RandomState(config.RANDOM_SEED)
    sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)
    X_sample = X_test[sample_idx]

    print(f"[explain] Building TreeExplainer for '{config.DEPLOYMENT_MODEL_NAME}'...")
    explainer = build_explainer(model)

    print(f"[explain] Computing global feature importance on {sample_size:,} sampled rows...")
    importance_df, shap_values = global_feature_importance(explainer, X_sample, feature_columns)
    print("\n[explain] Top 10 globally important features:")
    print(importance_df.head(10).to_string(index=False))

    importance_path = config.REPORTS_RESULTS_DIR / "shap_global_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"\n[saved] {importance_path}")

    summary_path = plot_summary(shap_values, X_sample, feature_columns)
    print(f"[saved figure] {summary_path}")

    # Dependence plots for a couple of features worth investigating for
    # interaction effects (e.g. a feature whose local SHAP contribution can
    # push a prediction in a direction opposite to its overall marginal
    # correlation with the label -- see reports/report_outline.md Section 14).
    for feature_of_interest in ("path_entropy", "number_of_dots"):
        if feature_of_interest in feature_columns:
            dep_path = plot_dependence(shap_values, X_sample, feature_columns, feature_of_interest)
            print(f"[saved figure] {dep_path}")

    # Demonstrate a local explanation on one example test-set URL.
    print("\n[explain] Example LOCAL explanation (first test-set row):")
    example_row = X_test[0]
    result = explain_single_prediction(explainer, model, example_row, feature_columns)
    print(f"   URL: {test_df.iloc[0]['original_url']}")
    print(f"   Prediction: {result['prediction']}")
    print(f"   Phishing probability: {result['phishing_probability']}")
    print("   Top contributing features:")
    for c in result["top_contributing_features"]:
        direction = "toward PHISHING" if c["shap_value"] > 0 else "toward BENIGN"
        print(f"      {c['feature']} = {c['value']:.4g}  (shap={c['shap_value']:.4f}, {direction})")

    print("\n[reminder] SHAP explains what influenced the MODEL'S prediction. "
          "A high-importance feature does not, by itself, prove a URL is "
          "malicious or safe.")
    print("\n[done] Phase 10 finished.")


if __name__ == "__main__":
    main()
