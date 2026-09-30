"""
predict.py -- PHASE 12: SINGLE-URL PREDICTION PIPELINE

This module ties together every earlier phase for exactly ONE URL at a time:
    normalize (Phase 2) -> extract features (Phase 3) -> load model ->
    predict -> SHAP explain (Phase 10)

It is used by app/streamlit_app.py so the app and this module always agree
on how a URL becomes a prediction. It is also runnable standalone from the
command line for quick manual checks without opening Streamlit.

REPRODUCIBILITY (Phase 12): this module loads feature_columns.json and
enforces that EXACT column order on every prediction, so the model always
sees features in the same order it was trained on.
"""

import argparse
import json

import joblib
import numpy as np

from src import config
from src.preprocessing import normalize_single_url
from src.feature_extraction import extract_features
from src.explain import build_explainer, explain_single_prediction


class PhishingURLPredictor:
    """Loads the deployment model once; reuse this object for many predictions.

    If config.USE_CALIBRATED_MODEL is True and a calibrated version of the
    deployment model exists (produced by `python -m src.calibrate`), that
    calibrated model is loaded instead of the raw one -- its predicted
    probabilities are corrected to better match real-world frequency (see
    src/calibrate.py for why this matters: class_weight="balanced" fixes the
    decision boundary but inflates the raw probability estimates). The
    predicted CLASS (phishing/benign) is unaffected either way; only the
    reported risk probability changes.
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or config.DEPLOYMENT_MODEL_NAME
        calibrated_path = config.CALIBRATED_MODEL_PATHS.get(model_name)
        raw_path = config.MODEL_PATHS[model_name]

        self.is_calibrated = False
        if config.USE_CALIBRATED_MODEL and calibrated_path and calibrated_path.exists():
            model_path = calibrated_path
            self.is_calibrated = True
        elif raw_path.exists():
            model_path = raw_path
        else:
            raise FileNotFoundError(
                f"Model '{model_name}' not found at {raw_path} (or its calibrated "
                f"version at {calibrated_path}). Run 'python -m src.train' first, "
                f"then optionally 'python -m src.calibrate'."
            )

        self.model_name = model_name
        self.model = joblib.load(model_path)  # used for predict_proba / class (may be calibrated)

        if not config.FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(
                f"{config.FEATURE_COLUMNS_PATH} not found. Run 'python -m src.train' first."
            )
        with open(config.FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

        # SHAP's TreeExplainer needs the raw tree model, not the calibration
        # wrapper (CalibratedClassifierCV isn't a tree). So SHAP explanations
        # always come from the underlying uncalibrated tree model, while the
        # reported probability/class still comes from self.model above.
        self.explainer = build_explainer(self._get_tree_model_for_shap(self.model))

    @staticmethod
    def _get_tree_model_for_shap(model):
        """
        Unwrap a CalibratedClassifierCV (if present) to get the tree model SHAP
        needs. Depending on scikit-learn version this can involve peeling off
        several wrapper layers: CalibratedClassifierCV -> _CalibratedClassifier
        -> FrozenEstimator -> the actual tree model. Only wrapper objects
        expose a singular ".estimator" attribute; real model classes
        (XGBClassifier, RandomForestClassifier, DecisionTreeClassifier,
        LogisticRegression) do not, so unwrapping stops naturally at the
        real model.
        """
        current = model
        if hasattr(current, "calibrated_classifiers_"):
            current = current.calibrated_classifiers_[0]
            if hasattr(current, "base_estimator"):  # older sklearn naming
                current = current.base_estimator
        while hasattr(current, "estimator"):
            current = current.estimator
        return current

    def predict(self, raw_url: str, top_n_features: int = 5) -> dict:
        """
        Full pipeline for one URL string. Returns a dict with the
        normalization info, all extracted features, and the SHAP explanation.
        Raises ValueError if the URL cannot be parsed (this is surfaced to
        the user in Streamlit rather than silently guessing).
        """
        norm = normalize_single_url(raw_url)
        if not norm["is_parseable"]:
            raise ValueError(
                f"Could not parse this URL into a hostname. "
                f"Original input: {raw_url!r}"
            )

        features = extract_features(norm["normalized_url"], norm["stripped_url"])

        # Build the feature vector in the EXACT saved column order.
        try:
            feature_vector = np.array(
                [features[col] for col in self.feature_columns], dtype=float
            )
        except KeyError as missing:
            raise RuntimeError(
                f"Feature column {missing} from training is missing from "
                f"extract_features() output. The feature extraction code has "
                f"drifted from what the model was trained on -- retrain the "
                f"model or fix feature_extraction.py."
            )

        explanation = explain_single_prediction(
            self.explainer, self.model, feature_vector, self.feature_columns,
            top_n=top_n_features,
        )

        return {
            "original_url": raw_url,
            "normalized_url": norm["normalized_url"],
            "model_used": self.model_name,
            "is_calibrated": self.is_calibrated,
            "prediction": explanation["prediction"],
            "phishing_probability": explanation["phishing_probability"],
            "top_contributing_features": explanation["top_contributing_features"],
            "all_features": features,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether a single URL is phishing or benign (static analysis only)."
    )
    parser.add_argument("url", type=str, help="The URL to analyze, e.g. https://example.com/login")
    args = parser.parse_args()

    predictor = PhishingURLPredictor()
    result = predictor.predict(args.url)

    calibration_note = "calibrated" if result["is_calibrated"] else "NOT calibrated -- run 'python -m src.calibrate'"
    print(f"\nURL:              {result['original_url']}")
    print(f"Model used:       {result['model_used']} ({calibration_note})")
    print(f"Prediction:       {result['prediction']}")
    print(f"Risk probability: {result['phishing_probability']}")
    print("\nTop contributing features:")
    for c in result["top_contributing_features"]:
        direction = "toward PHISHING" if c["shap_value"] > 0 else "toward BENIGN"
        print(f"   {c['feature']} = {c['value']:.4g}  ({direction}, shap={c['shap_value']:.4f})")

    print("\nThis tool performs static URL analysis only. "
          "It does not visit or verify the destination website.")


if __name__ == "__main__":
    main()