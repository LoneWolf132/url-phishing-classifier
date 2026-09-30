"""
test_features.py -- PHASE 4: UNIT TESTS

Run from the project root with:
    pytest tests/test_features.py -v

These tests use src.preprocessing.normalize_single_url() first (exactly as
the real pipeline does), then feed the result into
src.feature_extraction.extract_features(). This proves Phase 2 and Phase 3
agree with each other, not just that each works in isolation.
"""

import math

import pytest

from src.preprocessing import normalize_single_url
from src.feature_extraction import extract_features, shannon_entropy

TEST_URLS = [
    "https://example.com",
    "http://example.com/login",
    "http://192.168.1.1/login",
    "https://login.example.com/account/verify",
    "https://example.com/a/b/c?x=123&y=456",
    # Schemeless, matching the real dataset's dominant style:
    "nobell.it/70ffb52d079109dca5664cce6f317373782/login.SkyPe.com/",
]


def _features_for(url: str) -> dict:
    norm = normalize_single_url(url)
    assert norm["is_parseable"], f"URL failed to parse: {url}"
    return extract_features(norm["normalized_url"], norm["stripped_url"])


# ----------------------------------------------------------------------------
# General robustness: every test URL should produce a full numeric feature set
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("url", TEST_URLS)
def test_feature_extraction_does_not_crash(url):
    feats = _features_for(url)
    assert isinstance(feats, dict)
    assert len(feats) > 0


@pytest.mark.parametrize("url", TEST_URLS)
def test_features_are_numeric_where_expected(url):
    feats = _features_for(url)
    non_numeric_allowed = {"subdomain", "registered_domain", "suffix"}
    for key, value in feats.items():
        if key in non_numeric_allowed:
            continue
        assert isinstance(value, (int, float)), f"{key} is not numeric: {value!r}"


# ----------------------------------------------------------------------------
# Specific, hand-checkable expectations
# ----------------------------------------------------------------------------
def test_https_example_com_lengths():
    feats = _features_for("https://example.com")
    assert feats["hostname_length"] == len("example.com")
    assert feats["path_length"] == 0
    assert feats["has_https"] == 1


def test_path_depth_counts_segments():
    feats = _features_for("https://example.com/a/b/c?x=123&y=456")
    assert feats["path_depth"] == 3  # a, b, c
    assert feats["number_of_ampersands"] == 1
    assert feats["number_of_equals"] == 2
    assert feats["number_of_question_marks"] == 1


def test_root_path_has_zero_depth():
    feats = _features_for("http://example.com/login")
    assert feats["path_depth"] == 1  # "login"


def test_ip_address_detection():
    feats = _features_for("http://192.168.1.1/login")
    assert feats["has_ip_address"] == 1
    assert feats["registered_domain"] == ""  # an IP has no registrable domain


def test_non_ip_hostname_not_flagged_as_ip():
    feats = _features_for("https://example.com")
    assert feats["has_ip_address"] == 0


def test_subdomain_detection():
    feats = _features_for("https://login.example.com/account/verify")
    assert feats["number_of_subdomains"] == 1
    assert feats["subdomain"] == "login"
    assert feats["registered_domain"] == "example.com"


def test_suspicious_keyword_detection():
    feats = _features_for("https://login.example.com/account/verify")
    assert feats["has_suspicious_keyword"] == 1

    feats_clean = _features_for("https://example.com")
    assert feats_clean["has_suspicious_keyword"] == 0


def test_entropy_is_finite_and_non_negative():
    for url in TEST_URLS:
        feats = _features_for(url)
        for key in ("url_entropy", "hostname_entropy", "path_entropy"):
            value = feats[key]
            assert math.isfinite(value), f"{key} not finite for {url}"
            assert value >= 0, f"{key} negative for {url}"


def test_shannon_entropy_known_values():
    # Repeated single character -> zero entropy (fully predictable).
    assert shannon_entropy("aaaa") == 0.0
    # Empty string -> defined as zero entropy.
    assert shannon_entropy("") == 0.0
    # "aab": p(a)=2/3, p(b)=1/3 -> H = -(2/3*log2(2/3) + 1/3*log2(1/3)) ~= 0.918
    assert shannon_entropy("aab") == pytest.approx(0.918, abs=0.01)


def test_schemeless_url_still_extracts_hostname_features():
    # This is the dominant style in the real dataset (Phase 1 finding:
    # 99.98% of URLs have no scheme). Confirms Phase 2's scheme-adding logic
    # feeds Phase 3 correctly.
    feats = _features_for("nobell.it/70ffb52d079109dca5664cce6f317373782/login.SkyPe.com/")
    assert feats["hostname_length"] == len("nobell.it")
    assert feats["registered_domain"] == "nobell.it"
    assert feats["has_suspicious_keyword"] == 1  # "login" appears in the path


@pytest.mark.parametrize("url", [
    "example.com:xyz123/path",       # non-numeric "port"
    "weird-site.com:notaport/login", # non-numeric "port"
    "a.com:9999999999999/x",         # numeric but out of valid port range
])
def test_malformed_port_does_not_crash(url):
    # Regression test: urlparse().port raises ValueError on a colon
    # followed by text that isn't a valid port number (0-65535). Real
    # phishing/obfuscated URLs can contain exactly this pattern, so
    # extraction must degrade gracefully (has_port=0) instead of crashing.
    feats = _features_for(url)
    assert feats["has_port"] == 0


def test_ratios_stay_within_zero_and_one():
    for url in TEST_URLS:
        feats = _features_for(url)
        assert 0.0 <= feats["digit_ratio"] <= 1.0
        assert 0.0 <= feats["special_character_ratio"] <= 1.0
