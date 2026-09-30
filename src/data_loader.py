"""
data_loader.py -- PHASE 1: DATA INGESTION

Run from the project root with:
    python -m src.data_loader

It loads the raw CSV, inspects it, applies a small set of documented cleaning rules,
and saves data/processed/cleaned_urls.csv (columns: original_url, label).
"""

import json
from pathlib import Path

import pandas as pd

from src import config


# ----------------------------------------------------------------------------
# 1. Finding and loading the file
# ----------------------------------------------------------------------------
def find_raw_csv() -> Path:
    """Return the single CSV in data/raw/. Stop if there are zero or several."""
    csv_files = sorted(config.DATA_RAW_DIR.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No CSV file found in {config.DATA_RAW_DIR}. "
            "Download the dataset from Kaggle and place the CSV there."
        )
    if len(csv_files) > 1:
        names = [p.name for p in csv_files]
        raise RuntimeError(
            f"Found {len(csv_files)} CSV files in {config.DATA_RAW_DIR}: {names}. "
            "Keep exactly one so we never analyse the wrong file."
        )
    return csv_files[0]


def load_raw_csv(path: Path) -> pd.DataFrame:
    """
    Read the CSV with every column as text and NO automatic NA conversion.

    Why: by default pandas turns strings like "NA", "null" or "None" into missing
    values. A URL column could legitimately contain such text, so we read
    everything as plain strings and decide what counts as "missing" ourselves.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
            print(f"[load] Read '{path.name}' using encoding '{encoding}'.")
            return df
        except UnicodeDecodeError:
            print(f"[load] Encoding '{encoding}' failed, trying the next option...")
    raise RuntimeError("Could not read the file with any supported encoding.")


# ----------------------------------------------------------------------------
# 2. Identifying columns (never assume; stop if unsure)
# ----------------------------------------------------------------------------
def _find_column(df: pd.DataFrame, override, candidates, purpose: str) -> str:
    if override is not None:
        if override not in df.columns:
            raise ValueError(
                f"config override '{override}' for the {purpose} column is not in the "
                f"CSV. Columns found: {list(df.columns)}"
            )
        return override

    lowered = {str(col).strip().lower(): col for col in df.columns}
    matches = [lowered[name] for name in candidates if name in lowered]

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise ValueError(
            f"Could not identify the {purpose} column. Columns found: {list(df.columns)}. "
            f"Fix: set the matching *_OVERRIDE value in src/config.py."
        )
    raise ValueError(
        f"Ambiguous {purpose} column: several candidates matched {matches}. "
        f"Fix: set the matching *_OVERRIDE value in src/config.py."
    )


def identify_url_column(df: pd.DataFrame) -> str:
    return _find_column(df, config.URL_COLUMN_OVERRIDE, config.URL_COLUMN_CANDIDATES, "URL")


def identify_label_column(df: pd.DataFrame) -> str:
    return _find_column(df, config.LABEL_COLUMN_OVERRIDE, config.LABEL_COLUMN_CANDIDATES, "label")


# ----------------------------------------------------------------------------
# 3. Converting labels to 0 / 1
# ----------------------------------------------------------------------------
def convert_labels(raw_labels: pd.Series) -> pd.Series:
    """Map text labels to 0 (benign) / 1 (phishing). Stop on anything unrecognised."""
    normalized = raw_labels.str.strip().str.lower()
    unique_values = sorted(normalized.unique())
    print(f"[labels] Distinct label values found: {unique_values}")

    unknown = [v for v in unique_values if v not in config.LABEL_TEXT_MAP]
    if unknown:
        raise ValueError(
            f"Unrecognised label value(s): {unknown}. I refuse to guess which class "
            f"they mean (numeric 0/1 labels are especially dangerous to guess). "
            f"Tell me what they mean, then add them to LABEL_TEXT_MAP in src/config.py."
        )
    return normalized.map(config.LABEL_TEXT_MAP).astype(int)


def _print_class_table(labels: pd.Series, title: str) -> None:
    counts = labels.value_counts().sort_index()
    total = int(counts.sum())
    print(title)
    for value, count in counts.items():
        name = "benign" if value == 0 else "phishing"
        print(f"   {value} ({name}): {int(count):,}  ({count / total:.2%})")
    if len(counts) == 2:
        print(f"   imbalance ratio (majority : minority) = {counts.max() / counts.min():.2f} : 1")


# ----------------------------------------------------------------------------
# 4. Inspection + conservative cleaning
# ----------------------------------------------------------------------------
def inspect_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    report = {}

    url_col = identify_url_column(df)
    label_col = identify_label_column(df)
    print(f"[columns] URL column   = '{url_col}'")
    print(f"[columns] Label column = '{label_col}'")

    work = pd.DataFrame({"original_url": df[url_col], "raw_label": df[label_col]})
    n_start = len(work)
    report["rows_in_raw_file"] = n_start

    # ---- Missing values ----
    missing_url = work["original_url"].str.strip() == ""
    missing_label = work["raw_label"].str.strip() == ""
    print(f"\n[missing] Rows with empty URL:   {int(missing_url.sum()):,}")
    print(f"[missing] Rows with empty label: {int(missing_label.sum()):,}")
    report["rows_with_empty_url"] = int(missing_url.sum())
    report["rows_with_empty_label"] = int(missing_label.sum())

    # ---- Raw class counts (before removing anything) ----
    print("\n[classes] Raw label values before any removal:")
    print(work["raw_label"].value_counts().to_string())

    # ---- Read-only inspections (nothing is removed here) ----
    present = work.loc[~missing_url, "original_url"]
    stripped = present.str.strip()

    has_http = stripped.str.lower().str.match(r"^https?://")
    has_any_scheme = stripped.str.match(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
    n_present = len(present)
    print(f"\n[protocol] URLs starting with http:// or https://: "
          f"{int(has_http.sum()):,} ({has_http.mean():.2%})")
    print(f"[protocol] URLs starting with ANY scheme://:        "
          f"{int(has_any_scheme.sum()):,} ({has_any_scheme.mean():.2%})")
    print(f"[protocol] URLs with NO scheme at all:              "
          f"{int((~has_any_scheme).sum()):,} ({(~has_any_scheme).mean():.2%})")

    n_whitespace = int((present != stripped).sum())
    print(f"[whitespace] URLs with leading/trailing whitespace: {n_whitespace:,}")

    lengths = stripped.str.len()
    print(f"[length] min={int(lengths.min())}, median={int(lengths.median())}, "
          f"max={int(lengths.max())}, URLs longer than 2000 chars={int((lengths > 2000).sum())}")

    report["url_scheme_http_or_https"] = int(has_http.sum())
    report["url_no_scheme"] = int((~has_any_scheme).sum())
    report["url_with_edge_whitespace"] = n_whitespace
    report["url_length_min"] = int(lengths.min())
    report["url_length_median"] = int(lengths.median())
    report["url_length_max"] = int(lengths.max())

    # ---- CLEANING RULE 1: drop rows with empty URL or empty label ----
    drop_mask = missing_url | missing_label
    n_missing_dropped = int(drop_mask.sum())
    work = work[~drop_mask].copy()

    work["label"] = convert_labels(work["raw_label"])
    print()
    _print_class_table(work["label"], "[classes] After dropping empty rows:")

    # ---- Duplicate inspection ----
    n_exact_dup_rows = int(work.duplicated(subset=["original_url", "label"]).sum())
    print(f"\n[duplicates] Rows that repeat an earlier (URL, label) pair: {n_exact_dup_rows:,}")

    labels_per_url = work.groupby("original_url")["label"].nunique()
    conflicting_urls = labels_per_url[labels_per_url > 1].index
    print(f"[duplicates] Distinct URLs that appear with BOTH labels: {len(conflicting_urls):,}")

    # ---- CLEANING RULE 2: remove URLs with contradictory labels (saved separately) ----
    conflict_mask = work["original_url"].isin(conflicting_urls)
    conflicts = work.loc[conflict_mask, ["original_url", "label"]].sort_values("original_url")
    remaining = work[~conflict_mask]

    # ---- CLEANING RULE 3: drop exact duplicates, keep the first copy ----
    cleaned = remaining.drop_duplicates(subset=["original_url", "label"], keep="first")
    n_dupes_removed = len(remaining) - len(cleaned)
    cleaned = cleaned[["original_url", "label"]].reset_index(drop=True)

    # ---- Row accounting: every row must be explained ----
    n_conflict_rows = len(conflicts)
    n_final = len(cleaned)
    print("\n[accounting] Where did every row go?")
    print(f"   rows in raw file:                    {n_start:,}")
    print(f"   - dropped (empty URL/label):         {n_missing_dropped:,}")
    print(f"   - removed (conflicting labels):      {n_conflict_rows:,}")
    print(f"   - removed (exact duplicates):        {n_dupes_removed:,}")
    print(f"   = rows in cleaned dataset:           {n_final:,}")
    assert n_start == n_missing_dropped + n_conflict_rows + n_dupes_removed + n_final, \
        "Row accounting does not add up -- something is wrong in the cleaning logic."

    print()
    _print_class_table(cleaned["label"], "[classes] FINAL cleaned dataset:")

    # ---- Save outputs ----
    config.ensure_directories()
    cleaned.to_csv(config.CLEANED_DATASET_PATH, index=False)
    conflicts.to_csv(config.CONFLICTING_URLS_PATH, index=False)

    report.update({
        "rows_dropped_empty": n_missing_dropped,
        "rows_removed_conflicting_labels": n_conflict_rows,
        "rows_removed_exact_duplicates": n_dupes_removed,
        "rows_in_cleaned_dataset": n_final,
        "final_class_counts": {str(k): int(v) for k, v in cleaned["label"].value_counts().items()},
        "label_convention": "1 = phishing, 0 = benign",
        "url_column_used": url_col,
        "label_column_used": label_col,
    })
    with open(config.INSPECTION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[saved] {config.CLEANED_DATASET_PATH}")
    print(f"[saved] {config.CONFLICTING_URLS_PATH}")
    print(f"[saved] {config.INSPECTION_REPORT_PATH}")
    return cleaned


# ----------------------------------------------------------------------------
# 5. Helper for later phases
# ----------------------------------------------------------------------------
def load_cleaned_dataset() -> pd.DataFrame:
    """Reload cleaned_urls.csv safely (same no-auto-NA rule as the raw load)."""
    return pd.read_csv(
        config.CLEANED_DATASET_PATH,
        dtype={"original_url": str, "label": int},
        keep_default_na=False,
    )


def main() -> None:
    config.ensure_directories()
    csv_path = find_raw_csv()
    df = load_raw_csv(csv_path)

    print(f"\n[shape] {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[columns] {list(df.columns)}")
    print("\n[preview] First 5 rows:")
    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        print(df.head().to_string())

    print()
    inspect_and_clean(df)
    print("\n[done] Phase 1 finished.")


if __name__ == "__main__":
    main()