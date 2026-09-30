"""
preprocessing.py -- PHASE 2: URL NORMALIZATION

Run from the project root with:
    python -m src.preprocessing

Reads data/processed/cleaned_urls.csv (Phase 1 output) and produces:
    data/processed/normalized_urls.csv   -- usable rows
    data/processed/unparseable_urls.csv  -- rows we could not parse (kept, not deleted)

RULES (do not silently change these):
  - We do NOT visit the URL.
  - We do NOT resolve redirects.
  - We do NOT query DNS.
  - We do NOT lowercase or reorder anything.
  - We only strip surrounding whitespace and add a scheme (http://) when one
    is missing, purely so urllib.parse can split the URL into components.
"""

import re
from urllib.parse import urlparse

import pandas as pd

from src import config
from src.data_loader import load_cleaned_dataset

# Matches "something://" at the start, e.g. http://, https://, ftp://
_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def has_scheme(url: str) -> bool:
    """True if the URL already starts with some_scheme://."""
    return bool(_SCHEME_PATTERN.match(url))


def normalize_single_url(original_url: str) -> dict:
    """
    Turn one raw URL into the fields Phase 3 needs.

    Steps (in order):
      1. Strip surrounding whitespace only -> stripped_url.
      2. If stripped_url already has a scheme, normalized_url = stripped_url.
         Otherwise, normalized_url = "http://" + stripped_url (parsing aid only).
      3. Try urlparse() on normalized_url. If it yields no hostname (netloc),
         mark the row as unparseable instead of guessing.
    """
    stripped_url = original_url.strip()
    scheme_present = has_scheme(stripped_url)
    normalized_url = stripped_url if scheme_present else f"http://{stripped_url}"

    is_parseable = False
    try:
        parsed = urlparse(normalized_url)
        if parsed.netloc.strip() != "":
            is_parseable = True
    except ValueError:
        is_parseable = False

    return {
        "original_url": original_url,
        "stripped_url": stripped_url,
        "normalized_url": normalized_url,
        "has_scheme": scheme_present,
        "is_parseable": is_parseable,
    }


def process_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply normalize_single_url() to every row. Returns (normalized_df, rejected_df)."""
    records = [normalize_single_url(u) for u in df["original_url"]]
    result = pd.DataFrame(records)
    result["label"] = df["label"].values

    rejected = result[~result["is_parseable"]].copy()
    normalized = result[result["is_parseable"]].copy()
    normalized = normalized.drop(columns=["is_parseable"]).reset_index(drop=True)
    rejected = rejected.drop(columns=["is_parseable"]).reset_index(drop=True)

    return normalized, rejected


def _print_summary(df: pd.DataFrame, normalized: pd.DataFrame, rejected: pd.DataFrame) -> None:
    n_total = len(df)
    n_ok = len(normalized)
    n_rejected = len(rejected)

    print(f"[normalize] Input rows:                {n_total:,}")
    print(f"[normalize] Successfully normalized:   {n_ok:,} ({n_ok / n_total:.2%})")
    print(f"[normalize] Rejected (unparseable):    {n_rejected:,} ({n_rejected / n_total:.2%})")

    if n_ok > 0:
        n_had_scheme = int(normalized["has_scheme"].sum())
        n_added_scheme = n_ok - n_had_scheme
        print(f"\n[scheme] Already had a scheme:         {n_had_scheme:,} ({n_had_scheme / n_ok:.2%})")
        print(f"[scheme] Scheme added for parsing:     {n_added_scheme:,} ({n_added_scheme / n_ok:.2%})")

    if n_rejected > 0:
        print("\n[rejected] First 5 examples of rejected rows:")
        with pd.option_context("display.max_colwidth", 80):
            print(rejected[["original_url", "label"]].head().to_string())

    print("\n[classes] Class balance in normalized_urls.csv (should closely match Phase 1):")
    counts = normalized["label"].value_counts().sort_index()
    for value, count in counts.items():
        name = "benign" if value == 0 else "phishing"
        print(f"   {value} ({name}): {int(count):,} ({count / n_ok:.2%})")


def main() -> None:
    config.ensure_directories()
    df = load_cleaned_dataset()
    print(f"[load] Loaded {len(df):,} rows from {config.CLEANED_DATASET_PATH}")

    normalized, rejected = process_dataframe(df)
    _print_summary(df, normalized, rejected)

    normalized.to_csv(config.NORMALIZED_DATASET_PATH, index=False)
    rejected.to_csv(config.UNPARSEABLE_URLS_PATH, index=False)

    print(f"\n[saved] {config.NORMALIZED_DATASET_PATH}")
    print(f"[saved] {config.UNPARSEABLE_URLS_PATH}")
    print("\n[done] Phase 2 finished.")


if __name__ == "__main__":
    main()
