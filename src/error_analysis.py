"""
error_analysis.py -- PHASE 14: ERROR ANALYSIS

Run from the project root with:
    python -m src.error_analysis

Uses the deployment model (config.DEPLOYMENT_MODEL_NAME) on the held-out
test set and saves the actual misclassified URLs to:
    reports/results/false_positives.csv  -- benign URLs predicted phishing
    reports/results/false_negatives.csv  -- phishing URLs predicted benign

PURPOSE: the point of this phase is not to make the model look better --
it is to demonstrate understanding of WHERE and WHY a static, lexical-only
classifier fails. Good candidates to look for by hand in the saved CSVs:
    - long legitimate SaaS URLs (many query parameters, long paths)
    - OAuth redirect URLs (long, parameter-heavy, but legitimate)
    - URL shorteners (very short, low information content)
    - legitimate cloud platforms (e.g. storage buckets) hosting malicious content
    - lookalike domains (e.g. "exarnple.com" instead of "example.com")
    - high-entropy but benign URLs (e.g. auto-generated tracking links)
"""

import json

import joblib
import pandas as pd

from src import config
from src.train import get_feature_columns


def main() -> None:
    config.ensure_directories()

    model_path = config.MODEL_PATHS[config.DEPLOYMENT_MODEL_NAME]
    if not model_path.exists():
        print(f"[error] Model not found at {model_path}. Run 'python -m src.train' first.")
        return

    model = joblib.load(model_path)
    with open(config.FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    test_df = pd.read_csv(config.TEST_SPLIT_PATH)
    X_test = test_df[feature_columns].values
    y_true = test_df["label"].values

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_df = test_df.copy()
    test_df["predicted_label"] = y_pred
    test_df["predicted_probability"] = y_proba

    false_positives = test_df[(y_true == 0) & (y_pred == 1)].copy()
    false_negatives = test_df[(y_true == 1) & (y_pred == 0)].copy()

    false_positives = false_positives.sort_values("predicted_probability", ascending=False)
    false_negatives = false_negatives.sort_values("predicted_probability", ascending=True)

    print(f"[error_analysis] Test set size: {len(test_df):,}")
    print(f"[error_analysis] False positives (benign predicted as phishing): {len(false_positives):,}")
    print(f"[error_analysis] False negatives (phishing predicted as benign): {len(false_negatives):,}")

    if len(false_positives) > 0:
        print("\n[error_analysis] Sample false positives (most confidently wrong first):")
        with pd.option_context("display.max_colwidth", 70):
            print(false_positives[["original_url", "predicted_probability"]].head(5).to_string(index=False))

    if len(false_negatives) > 0:
        print("\n[error_analysis] Sample false negatives (most confidently wrong first):")
        with pd.option_context("display.max_colwidth", 70):
            print(false_negatives[["original_url", "predicted_probability"]].head(5).to_string(index=False))

    false_positives.to_csv(config.FALSE_POSITIVES_PATH, index=False)
    false_negatives.to_csv(config.FALSE_NEGATIVES_PATH, index=False)
    print(f"\n[saved] {config.FALSE_POSITIVES_PATH}")
    print(f"[saved] {config.FALSE_NEGATIVES_PATH}")

    print(
        "\n[reminder] Inspect these files by hand. Look for patterns such as: "
        "long legitimate SaaS/OAuth URLs (often false positives), URL "
        "shorteners or lookalike domains (often false negatives), and "
        "high-entropy but benign auto-generated links. Write what you find "
        "into Section 14 (Error Analysis) of the academic report."
    )
    print("\n[done] Phase 14 finished.")


if __name__ == "__main__":
    main()
