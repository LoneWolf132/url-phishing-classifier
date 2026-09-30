"""
feature_extraction.py -- PHASE 3: FEATURE EXTRACTION

Run from the project root with:
    python -m src.feature_extraction

Turns every normalized URL into a numeric feature vector. Every feature is
computed by its own small function so each one can be tested and explained
independently. Nothing here visits the URL, resolves DNS, or makes network
requests -- everything is pure string/character analysis.
"""

import math
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import tldextract

from src import config

# tldextract normally tries to download an up-to-date public-suffix list from
# the internet on first use. Network requests are forbidden in this project,
# so we force it to use ONLY the copy bundled with the library.
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

VOWELS = set("aeiouAEIOU")


# ----------------------------------------------------------------------------
# SECTION 1: Basic length features
# ----------------------------------------------------------------------------
def length_features(parsed) -> dict:
    """
    Character counts of each URL component.
    WHY: phishing URLs are often abnormally long (to hide the real domain
    behind a long path) or use an unusually long/short hostname.
    """
    return {
        "url_length": len(parsed.geturl()),
        "hostname_length": len(parsed.hostname) if parsed.hostname else 0,
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
        "fragment_length": len(parsed.fragment),
    }


# ----------------------------------------------------------------------------
# SECTION 2: Structural features
# ----------------------------------------------------------------------------
def structural_features(parsed) -> dict:
    """
    Counts of structural/delimiter characters.
    WHY: phishing URLs often pack in extra subdomains, dots, and symbols to
    imitate a trusted brand name (e.g. "login.example.com.verify-account.ru").
    """
    full_url = parsed.geturl()
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    # path_depth: number of non-empty segments in the path.
    path_depth = len([seg for seg in path.split("/") if seg != ""])

    # number_of_subdomains: dot-separated hostname labels minus the last two
    # (domain + suffix). Never negative.
    host_labels = [lbl for lbl in hostname.split(".") if lbl != ""]
    number_of_subdomains = max(len(host_labels) - 2, 0)

    return {
        "path_depth": path_depth,
        "number_of_subdomains": number_of_subdomains,
        "number_of_dots": full_url.count("."),
        "number_of_hyphens": full_url.count("-"),
        "number_of_underscores": full_url.count("_"),
        "number_of_slashes": full_url.count("/"),
        "number_of_question_marks": full_url.count("?"),
        "number_of_equals": full_url.count("="),
        "number_of_ampersands": full_url.count("&"),
        "number_of_at_symbols": full_url.count("@"),
        "number_of_percent_symbols": full_url.count("%"),
    }


# ----------------------------------------------------------------------------
# SECTION 3: Character features
# ----------------------------------------------------------------------------
def character_features(parsed) -> dict:
    """
    Composition of the URL string: digits vs letters vs symbols vs vowels.
    WHY: phishing URLs often contain long random-looking digit/hex strings
    (session IDs, tracking codes) that legitimate short URLs rarely have.
    """
    full_url = parsed.geturl()
    length = len(full_url) if len(full_url) > 0 else 1  # avoid divide-by-zero

    digit_count = sum(c.isdigit() for c in full_url)
    alphabetic_count = sum(c.isalpha() for c in full_url)
    special_character_count = length - digit_count - alphabetic_count
    vowel_count = sum(c in VOWELS for c in full_url)
    consonant_count = sum(c.isalpha() and c not in VOWELS for c in full_url)

    return {
        "digit_count": digit_count,
        "digit_ratio": digit_count / length,
        "alphabetic_count": alphabetic_count,
        "special_character_count": special_character_count,
        "special_character_ratio": special_character_count / length,
        "vowel_count": vowel_count,
        "consonant_count": consonant_count,
        # +1 in denominator avoids divide-by-zero when there are 0 consonants.
        "vowel_consonant_ratio": vowel_count / (consonant_count + 1),
    }


# ----------------------------------------------------------------------------
# SECTION 4: Security / heuristic features
# ----------------------------------------------------------------------------
def _is_ip_address(hostname: str) -> bool:
    """
    True if the hostname is a bare IPv4 address, e.g. "192.168.1.1".
    A pure string check -- no socket/DNS calls, no network activity.
    """
    parts = hostname.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        if not (0 <= int(part) <= 255):
            return False
    return True


def _safe_has_port(parsed) -> int:
    """
    parsed.port does not just read text after a colon -- it validates that
    the text is a real port number (0-65535) and RAISES ValueError if it
    isn't (e.g. a hostname like "example.com:xyz123" from a malformed or
    deliberately obfuscated URL). Real-world phishing/URL datasets contain
    exactly this kind of malformed netloc, so we treat "could not be
    determined as a valid port" as its own safe case (0) rather than
    letting the whole pipeline crash on one unusual row.
    """
    try:
        return int(parsed.port is not None)
    except ValueError:
        return 0


def security_heuristic_features(parsed, stripped_url: str) -> dict:
    """
    Simple heuristic flags used in phishing-detection literature.
    These are fed to the model as FEATURES ONLY -- never treated as proof of
    phishing on their own.
    """
    hostname = parsed.hostname or ""
    full_url = parsed.geturl()
    lower_url = full_url.lower()

    return {
        "has_ip_address": int(_is_ip_address(hostname)),
        "has_https": int(parsed.scheme == "https"),
        # "@" lets browsers ignore everything before it -- a classic trick
        # to disguise the real destination.
        "has_at_symbol": int("@" in stripped_url),
        "has_percent_encoding": int("%" in full_url),
        "has_port": _safe_has_port(parsed),
        "has_suspicious_keyword": int(
            any(keyword in lower_url for keyword in config.SUSPICIOUS_KEYWORDS)
        ),
    }


# ----------------------------------------------------------------------------
# SECTION 5: Entropy
# ----------------------------------------------------------------------------
def shannon_entropy(text: str) -> float:
    """
    Shannon entropy, in bits, of a piece of text.

    Formula:  H(X) = - sum( p(c) * log2(p(c)) )  for each distinct character c,
    where p(c) is how often c appears in the text (its frequency / length).

    Intuition: a string using few characters repeated a lot (e.g. "aaaaaa")
    has LOW entropy -- predictable. A string using many different characters
    with no repeating pattern (e.g. a random session token) has HIGH entropy.
    Phishing URLs sometimes use randomised, high-entropy paths or hostnames
    to generate unique, hard-to-blocklist links.

    Empty text has no characters to measure, so entropy is defined as 0.0.
    """
    if len(text) == 0:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return entropy


def entropy_features(parsed) -> dict:
    return {
        "url_entropy": shannon_entropy(parsed.geturl()),
        "hostname_entropy": shannon_entropy(parsed.hostname or ""),
        "path_entropy": shannon_entropy(parsed.path or ""),
    }


# ----------------------------------------------------------------------------
# SECTION 6: Domain features (tldextract)
# ----------------------------------------------------------------------------
def domain_features(hostname: str) -> dict:
    """
    Splits a hostname into subdomain / registered domain / suffix using
    tldextract. Example: "login.example.co.uk" ->
        subdomain          = "login"
        registered_domain  = "example.co.uk"   (the eTLD+1)
        suffix             = "co.uk"

    registered_domain is used again in Phase 5 as the grouping key that
    prevents data leakage between train and test sets.
    """
    extracted = _TLD_EXTRACTOR(hostname)
    subdomain = extracted.subdomain
    # tldextract renamed "registered_domain" to "top_domain_under_public_suffix"
    # in newer versions. Try the new name first, fall back to the old one so
    # this works across tldextract versions without a deprecation warning.
    if hasattr(extracted, "top_domain_under_public_suffix"):
        registered_domain = extracted.top_domain_under_public_suffix
    else:
        registered_domain = extracted.registered_domain
    suffix = extracted.suffix

    return {
        "subdomain": subdomain,
        "registered_domain": registered_domain,
        "suffix": suffix,
        "registered_domain_length": len(registered_domain),
        "subdomain_length": len(subdomain),
    }


# ----------------------------------------------------------------------------
# MASTER FUNCTION: one URL in, one feature dict out
# ----------------------------------------------------------------------------
def extract_features(normalized_url: str, stripped_url: str) -> dict:
    """
    Combine every feature category for a single (already-normalized) URL.

    normalized_url: the version WITH a scheme (used for parsing).
    stripped_url:   the version as originally written (used for
                     has_at_symbol, so an @ that was really typed is
                     detected honestly, independent of scheme-adding).
    """
    parsed = urlparse(normalized_url)

    features = {}
    features.update(length_features(parsed))
    features.update(structural_features(parsed))
    features.update(character_features(parsed))
    features.update(security_heuristic_features(parsed, stripped_url))
    features.update(entropy_features(parsed))
    features.update(domain_features(parsed.hostname or ""))
    return features


# ----------------------------------------------------------------------------
# Batch processing over the whole dataset
# ----------------------------------------------------------------------------
def build_feature_table() -> pd.DataFrame:
    df = pd.read_csv(config.NORMALIZED_DATASET_PATH, dtype=str, keep_default_na=False)
    df["label"] = df["label"].astype(int)

    print(f"[extract] Extracting features for {len(df):,} URLs...")
    records = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        feats = extract_features(row.normalized_url, row.stripped_url)
        feats["original_url"] = row.original_url
        feats["label"] = row.label
        records.append(feats)
        if i % 100_000 == 0:
            print(f"   ...{i:,} / {len(df):,} done")

    feature_df = pd.DataFrame(records)
    id_cols = ["original_url", "label", "registered_domain", "subdomain", "suffix"]
    other_cols = [c for c in feature_df.columns if c not in id_cols]
    feature_df = feature_df[id_cols + other_cols]

    print(f"[extract] Done. Feature table shape: {feature_df.shape}")
    return feature_df


def main() -> None:
    config.ensure_directories()
    feature_df = build_feature_table()
    feature_df.to_csv(config.FEATURES_DATASET_PATH, index=False)
    print(f"[saved] {config.FEATURES_DATASET_PATH}")
    print("\n[done] Phase 3 finished.")


if __name__ == "__main__":
    main()
