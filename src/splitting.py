"""
splitting.py -- PHASE 5: DATA LEAKAGE PREVENTION

Run from the project root with:
    python -m src.splitting

WHY THIS PHASE EXISTS (read this before anything else):
If we used a plain random train_test_split on individual URLs, the SAME
registered domain (e.g. "example.com") could easily end up with some of its
URLs in the training set and others in the test set. A model can then learn
domain-specific quirks (a particular tracking parameter, a particular path
naming style used only by that one site) during training, and "recognise"
those exact same quirks in the test set -- not because it generalised, but
because it effectively saw that domain already. This produces an
artificially high test score that will NOT hold up on genuinely unseen
domains in the real world.

THE FIX: we split by registered_domain, not by URL. Every URL belonging to
the same registered domain goes ENTIRELY into train OR ENTIRELY into test,
never both. We use scikit-learn's GroupShuffleSplit with the group defined
as registered_domain (computed in Phase 3 via tldextract).

We then explicitly print and verify that the set of train domains and the
set of test domains have ZERO overlap. This is not a formality -- it is the
single most important safety check in the whole project.

Note on timestamps: this dataset has no timestamp column, so we cannot also
do a chronological split. If a future version of the dataset includes
timestamps, splitting by time in addition to domain would be an even
stronger test of generalisation (train on older URLs, test on newer ones,
since phishing campaigns evolve over time). We do not invent timestamps
here because that would misrepresent the dataset.
"""

import json

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config


def load_feature_table() -> pd.DataFrame:
    df = pd.read_csv(config.FEATURES_DATASET_PATH)
    return df


def _rows_with_empty_registered_domain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows where tldextract could not find a registered domain (e.g. the
    hostname is a bare IP address, like 192.168.1.1). These have no
    meaningful "domain" to group by, so each such URL is treated as its
    own group (using the full hostname/URL as a stand-in group id) rather
    than being silently dropped or lumped together into one giant group.
    """
    return df[df["registered_domain"].isna() | (df["registered_domain"] == "")]


def build_group_column(df: pd.DataFrame) -> pd.Series:
    """
    Returns the grouping key for each row: registered_domain when available,
    otherwise the row's own original_url (so IP-hosted or unusual URLs each
    form their own singleton group instead of being merged incorrectly).
    """
    empty_mask = df["registered_domain"].isna() | (df["registered_domain"] == "")
    group = df["registered_domain"].astype(str).copy()
    group[empty_mask] = "NOGROUP::" + df.loc[empty_mask, "original_url"].astype(str)
    return group


def leakage_safe_split(df: pd.DataFrame):
    """
    Split df into train/test using GroupShuffleSplit, grouped by domain.
    Returns (train_df, test_df, report_dict).
    """
    groups = build_group_column(df)

    n_no_domain = int((groups.str.startswith("NOGROUP::")).sum())
    if n_no_domain > 0:
        print(f"[split] {n_no_domain:,} rows have no registered domain "
              f"(e.g. IP-hosted URLs). Each is treated as its own group.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    intersection = train_groups & test_groups

    print(f"\n[split] Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")
    print(f"[split] Unique train domains/groups: {len(train_groups):,}")
    print(f"[split] Unique test domains/groups:  {len(test_groups):,}")
    print(f"[split] Intersection between train and test groups: {len(intersection):,}")

    assert len(intersection) == 0, (
        "DATA LEAKAGE DETECTED: some domain/group appears in both train and "
        "test sets. This should be impossible with GroupShuffleSplit -- stop "
        "and investigate before training any model."
    )
    print("[split] VERIFIED: zero overlap between train and test domains.")

    def _class_balance(d: pd.DataFrame) -> dict:
        counts = d["label"].value_counts().sort_index()
        total = len(d)
        return {
            "benign": int(counts.get(0, 0)),
            "phishing": int(counts.get(1, 0)),
            "phishing_ratio": round(counts.get(1, 0) / total, 4) if total else None,
        }

    report = {
        "test_size_target": config.TEST_SIZE,
        "random_seed": config.RANDOM_SEED,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_unique_groups": len(train_groups),
        "test_unique_groups": len(test_groups),
        "group_intersection_count": len(intersection),
        "rows_with_no_registered_domain": n_no_domain,
        "train_class_balance": _class_balance(train_df),
        "test_class_balance": _class_balance(test_df),
    }
    return train_df, test_df, report


def main() -> None:
    config.ensure_directories()
    print(f"[load] Reading {config.FEATURES_DATASET_PATH}")
    df = load_feature_table()
    print(f"[load] Feature table shape: {df.shape}")

    train_df, test_df, report = leakage_safe_split(df)

    print("\n[balance] Train class balance:", report["train_class_balance"])
    print("[balance] Test class balance: ", report["test_class_balance"])

    train_df.to_csv(config.TRAIN_SPLIT_PATH, index=False)
    test_df.to_csv(config.TEST_SPLIT_PATH, index=False)
    with open(config.SPLIT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[saved] {config.TRAIN_SPLIT_PATH}")
    print(f"[saved] {config.TEST_SPLIT_PATH}")
    print(f"[saved] {config.SPLIT_REPORT_PATH}")
    print("\n[done] Phase 5 finished.")


if __name__ == "__main__":
    main()
