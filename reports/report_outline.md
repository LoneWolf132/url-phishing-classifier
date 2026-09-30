# Structural Lexical Analysis and Machine Learning Architectures for Phishing URL Classification

**Report outline (Phase 13).** Fill in each section using YOUR actual numbers from
`reports/results/` and `reports/figures/` once you have run the full pipeline on
your machine. Do not invent citations, dataset statistics, or experimental
results that did not come from your own run.

---

## 1. Abstract
One paragraph: problem, approach, dataset, models compared, headline result
(e.g. best PR-AUC/F1), and the central limitation (static analysis only).

## 2. Introduction
What phishing URLs are, why URL-only detection is a useful complementary
layer to content- or reputation-based detection, and what this project builds.

## 3. Problem Statement
Binary classification: given a URL string alone (no page content, no live
lookups), predict phishing vs. benign.

## 4. Motivation
Why lexical/structural signals are attractive: no network cost, works before
a page even loads, works even if the destination is later taken down.

## 5. Literature Review
Summarize (in your own words, properly cited) prior work on URL-based
phishing detection, lexical/structural feature engineering, and the general
literature on the importance of leakage-safe evaluation in security ML.
**Do not fabricate citations** -- only reference sources you have actually read.

## 6. Dataset
- Source: Kaggle "Phishing Site URLs" (taruntiwarihp).
- **Use your own Phase 1 inspection numbers here**, not the numbers assumed
  before inspection: raw row count, class counts (`good`/`bad`), rows with
  no scheme, duplicate/conflicting-label rows removed, and the final class
  balance from `reports/results` (or wherever your Phase 1 output landed).
- Note the dataset has no timestamp column (relevant to Section 9).

## 7. Methodology
Walk through the pipeline stages in order: ingestion -> normalization ->
feature extraction -> grouped split -> model training -> evaluation ->
explainability. Reference the corresponding `src/` module for each stage.

## 8. Feature Engineering
List the six feature categories (length, structural, character, security/
heuristic, entropy, domain) and briefly justify 2-3 features per category.
Include the Shannon entropy formula and a short worked example (see the
docstring in `src/feature_extraction.py`).

## 9. Data Leakage Prevention
Explain, in your own words, why random URL-level splitting would let the
same registered domain appear in both train and test, artificially inflating
scores. Report your actual Phase 5 numbers: unique train domains, unique
test domains, and the domain intersection (must be reported as 0). Mention
that the dataset lacked timestamps, so chronological splitting was discussed
as a concept but not implemented, to avoid inventing data that does not exist.

## 10. Machine Learning Models
Describe the four models (Logistic Regression, Decision Tree, Random Forest,
XGBoost) and why each was included (linear baseline -> nonlinear baseline ->
classical ensemble -> gradient boosting).

## 11. Evaluation Metrics
Define each metric used (accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
MCC, FPR, FNR) and explain briefly why accuracy alone is misleading here
given the class imbalance found in Section 6.

## 12. Results
Insert your actual `reports/results/evaluation_results.json` numbers as a
table, and embed the figures from `reports/figures/` (confusion matrices,
ROC curves, PR curves, feature importance plots). State which model you
selected as the deployment candidate (`config.DEPLOYMENT_MODEL_NAME`) and
why, based on the evidence -- not because it is the most complex model.

## 13. SHAP Explainability
Insert the global feature importance table/plot from
`reports/results/shap_global_importance.csv` and
`reports/figures/shap_summary_plot.png`. Include one worked local-explanation
example (a specific URL, its prediction, and its top contributing features).
Restate explicitly: SHAP explains the model's reasoning, not ground truth
about whether a URL is actually malicious.

## 14. Error Analysis
Using `reports/results/false_positives.csv` and `false_negatives.csv`
(produced by `src/error_analysis.py`), discuss real examples you found of:
long legitimate SaaS/OAuth URLs misclassified as phishing, URL shorteners or
lookalike domains misclassified as benign, and any high-entropy-but-benign
URLs. Use only examples that actually appear in your generated CSVs.

## 15. Limitations
Static lexical analysis cannot see page content, cannot verify domain age
or reputation, cannot detect content injected after normalization, and can
be evaded by sufficiently URL-legitimate-looking phishing pages or by
legitimate platforms being abused for hosting. The class imbalance and the
single-dataset source also limit how far results generalize.

## 16. Conclusion
Answer the central research question directly and honestly, based on your
actual results:
> "Can structural and lexical properties of a URL alone provide an
> effective, lightweight signal for phishing URL classification without
> relying on live webpage content or external threat-intelligence queries?"
Do not claim the classifier "solves" phishing detection.

## 17. References
List only sources you actually consulted.
