# Structural Lexical Analysis and Machine Learning Architectures for Phishing URL Classification

A static, offline URL classifier. It predicts whether a URL is **phishing** or **benign**
using only the URL's text structure (length, symbols, entropy, domain structure, etc.).
**It never visits the URL, resolves DNS, or makes any network request to the destination.**

## What this project does

```
URL string
   -> validation / normalization           (src/preprocessing.py)
   -> structural + lexical feature extraction   (src/feature_extraction.py)
   -> domain-level grouping                 (tldextract, inside feature_extraction.py)
   -> leakage-safe train/test split         (src/splitting.py)
   -> ML model training (4 models, tuned)   (src/train.py)
   -> evaluation                            (src/evaluate.py)
   -> SHAP explainability                   (src/explain.py)
   -> Streamlit interface                   (app/streamlit_app.py)
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Get the dataset

Download **Kaggle - Phishing Site URLs**
(https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls),
unzip it, and place the single CSV file in `data/raw/`.

## 3. Run every phase, in order

Run each command from the project root, with the virtual environment active.

​```bash
# Phase 1 -- inspect and clean the raw dataset
python -m src.data_loader

# Phase 2 -- normalize URLs (safe, offline; adds a scheme only for parsing)
python -m src.preprocessing

# Phase 3 -- extract the full numeric feature set for every URL
python -m src.feature_extraction

# Phase 4 -- run the unit tests (does not need the dataset)
python -m pytest tests/test_features.py -v

# Phase 5 -- leakage-safe train/test split (grouped by registered domain)
python -m src.splitting

# Phases 6-8 -- train 4 models with class-weighting and a small tuned grid
python -m src.train
#   add --quick for a fast smoke test with a smaller hyperparameter grid:
python -m src.train --quick

# Phase 9 -- evaluate all 4 models on the held-out test set
python -m src.evaluate

# Phase 9 (extension) -- calibrate predicted probabilities
# WHY: class-weighting (used above to handle imbalance) fixes the decision
# boundary but inflates the raw probability outputs -- confirmed by the
# calibration curve from Phase 9. This step corrects that, without
# retraining or changing the underlying models' decisions.
python -m src.calibrate

# Phase 10 -- SHAP global + local explainability (also saves dependence plots
# that show feature INTERACTIONS, not just marginal importance)
python -m src.explain

# Phase 14 -- error analysis (false positives / false negatives)
python -m src.error_analysis

# Phase 11 -- launch the interactive app (uses the calibrated model
# automatically if step above was run; see config.USE_CALIBRATED_MODEL)
streamlit run app/streamlit_app.py
​```

You can also predict a single URL from the command line without opening the app:

​```bash
python -m src.predict "http://login.example.com/account/verify?id=12345"
​```
The output tells you whether the calibrated or raw model was used.

## 4. Project structure

```
phishing_url_classifier/
├── data/
│   ├── raw/            <- put the downloaded Kaggle CSV here
│   └── processed/      <- cleaned/normalized/feature/split CSVs (generated)
├── src/
│   ├── config.py            single source of truth: paths, seed, label rules
│   ├── data_loader.py       Phase 1: ingestion + cleaning
│   ├── preprocessing.py     Phase 2: normalization
│   ├── feature_extraction.py Phase 3: feature engineering
│   ├── splitting.py         Phase 5: leakage-safe grouped split
│   ├── train.py             Phases 6-8: models, class weighting, tuning
│   ├── evaluate.py          Phase 9: metrics, figures, comparison table
│   ├── explain.py           Phase 10: SHAP explainability
│   ├── error_analysis.py    Phase 14: false positive/negative inspection
│   └── predict.py           Phase 12: single-URL end-to-end pipeline
├── models/              <- trained .joblib models + feature_columns.json (generated)
├── reports/
│   ├── figures/         <- confusion matrices, ROC/PR curves, SHAP plots (generated)
│   ├── results/         <- JSON/CSV metrics, split report, error analysis (generated)
│   └── report_outline.md  Phase 13: academic report structure
├── app/
│   └── streamlit_app.py Phase 11: interactive web interface
├── tests/
│   └── test_features.py Phase 4: unit tests for feature extraction
├── requirements.txt
└── README.md
```

## 5. Design decisions worth knowing for a viva

- **Label convention:** `0` = benign, `1` = phishing, everywhere in the project.
- **No SMOTE:** class imbalance is handled with `class_weight="balanced"`
  (or XGBoost's `scale_pos_weight`), not synthetic oversampling. See the
  docstring at the top of `src/train.py` for the reasoning.
- **Leakage prevention:** the train/test split groups by `registered_domain`
  (via `tldextract`), using `GroupShuffleSplit`, and asserts zero domain
  overlap between train and test. Hyperparameter tuning also uses
  `GroupKFold` internally for the same reason.
- **No network calls anywhere:** `tldextract` is configured with
  `suffix_list_urls=()` so it never tries to download its suffix list; no
  module in `src/` makes an HTTP request, resolves DNS, or checks WHOIS.
- **Evaluation metric choice:** PR-AUC (average precision) is used for
  hyperparameter tuning and highlighted alongside ROC-AUC, because it is
  more informative than accuracy or ROC-AUC alone under class imbalance.
- **SHAP caveat:** SHAP explains what influenced the *model's* prediction,
  not proof that a feature makes a URL malicious. This is stated explicitly
  in the Streamlit app and in `src/explain.py`.

## 6. Known limitations (see also Section 15 of `reports/report_outline.md`)

Static lexical classification cannot see webpage content, cannot check
whether a domain was very recently registered, and can be fooled by
legitimate-looking long SaaS/OAuth URLs or lookalike domains. See
`src/error_analysis.py` and Section 14 of the report outline.
