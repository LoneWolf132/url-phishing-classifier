"""
streamlit_app.py -- PHASE 11: STREAMLIT APPLICATION

Run from the project root with:
    streamlit run app/streamlit_app.py

This app performs STATIC analysis only:
    normalize -> extract features -> load trained model -> predict -> explain

It NEVER makes a network request to the URL the user enters. Only the
project's own already-trained model files and the standard-library URL
parser are used.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow "from src...." imports when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import PhishingURLPredictor  # noqa: E402
from src import config  # noqa: E402


st.set_page_config(page_title="Phishing URL Classifier", page_icon="🔍", layout="centered")


@st.cache_resource
def get_predictor():
    """
    Loaded once per Streamlit server process (not on every button click),
    since loading the model and building the SHAP explainer has some cost.
    """
    return PhishingURLPredictor()


def main():
    st.title("Phishing URL Classifier")
    st.warning(
        "⚠️ This tool performs static URL analysis only. "
        "It does not visit or verify the destination website."
    )

    st.markdown(
        "Enter a URL below. It will be analyzed using only its **text structure** "
        "(length, symbols, entropy, domain structure, etc.) -- the destination "
        "is never contacted."
    )

    url_input = st.text_input("Enter a URL", placeholder="e.g. http://example.com/login")
    analyze_clicked = st.button("Analyze URL", type="primary")

    if not analyze_clicked:
        st.stop()

    if not url_input or not url_input.strip():
        st.error("Please enter a URL before clicking Analyze.")
        st.stop()

    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        st.error(
            f"Model files not found. Run the training pipeline first "
            f"(`python -m src.train`) before using this app.\n\nDetails: {e}"
        )
        st.stop()

    try:
        result = predictor.predict(url_input.strip())
    except ValueError as e:
        st.error(f"Could not analyze this URL: {e}")
        st.stop()
    except RuntimeError as e:
        st.error(f"Internal error: {e}")
        st.stop()

    # ---- 1. Prediction + probability ----
    st.subheader("Result")
    prediction = result["prediction"]
    probability = result["phishing_probability"]

    if prediction == "PHISHING":
        st.error(f"**Prediction: {prediction}**")
    else:
        st.success(f"**Prediction: {prediction}**")
    st.metric("Phishing risk probability", f"{probability:.2%}")
    st.progress(min(max(probability, 0.0), 1.0))

    calibration_tag = "calibrated" if result["is_calibrated"] else "not calibrated"
    st.caption(
        f"Model used: `{result['model_used']}` ({calibration_tag})  |  "
        f"Normalized URL analyzed: `{result['normalized_url']}`"
    )
    if not result["is_calibrated"]:
        st.info(
            "This model's raw probability output may be inflated relative to real-world "
            "frequency (a known side effect of class weighting used to handle imbalance). "
            "Run `python -m src.calibrate` to correct this.",
            icon="ℹ️",
        )

    # ---- 2. Top contributing features (SHAP) ----
    st.subheader("What influenced this prediction")
    st.caption(
        "SHAP explains what influenced the MODEL'S prediction. "
        "These features do not, on their own, prove a URL is malicious or safe."
    )
    contrib_df = pd.DataFrame(result["top_contributing_features"])
    contrib_df["direction"] = contrib_df["shap_value"].apply(
        lambda v: "→ Phishing" if v > 0 else "→ Benign"
    )
    contrib_df = contrib_df.rename(columns={
        "feature": "Feature", "value": "Value", "shap_value": "SHAP value",
        "direction": "Pushed prediction",
    })
    st.dataframe(
        contrib_df[["Feature", "Value", "SHAP value", "Pushed prediction"]],
        use_container_width=True,
        hide_index=True,
    )

    # ---- 3. Selected extracted features (full picture, collapsed by default) ----
    with st.expander("See all extracted features"):
        all_feats = result["all_features"]
        feats_df = pd.DataFrame(
            [{"Feature": k, "Value": v} for k, v in all_feats.items()]
        )
        st.dataframe(feats_df, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "This tool performs static URL analysis only. "
        "It does not visit or verify the destination website."
    )


if __name__ == "__main__":
    main()
