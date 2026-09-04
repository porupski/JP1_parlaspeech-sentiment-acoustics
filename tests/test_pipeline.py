# ============================================================
# Script:  test_pipeline.py
# Release: 1.0
# Version: v1.00
# Purpose: Basic sanity tests for core utilities.
#          Run with: pytest tests/
# ============================================================

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_utils import (score_to_label, score_to_bin, filter_word_count,
                                filter_speakers_by_coverage, SENTIMENT_LABELS)
from utils.stats import (matched_pairs_rbc, concordance_prob, bh_correct,
                          bootstrap_ci, h3_check)
from utils.extraction import count_syllables


# ---------------------------------------------------------------------------
# data_utils
# ---------------------------------------------------------------------------

def test_score_to_label():
    assert score_to_label(0.0) == "Negative"
    assert score_to_label(2.5) == "Neutral Negative"
    assert score_to_label(5.0) == "Positive"


def test_score_to_bin_boundaries():
    assert score_to_bin(0.0) == 0
    assert score_to_bin(5.0) == 59
    assert score_to_bin(2.5) == 30


def test_filter_word_count():
    df = pd.DataFrame({"n_words": [5, 10, 25, 40, 45]})
    out = filter_word_count(df, min_w=10, max_w=40)
    assert list(out["n_words"]) == [10, 25, 40]


def test_filter_speakers_by_coverage():
    rows = []
    for spk in ["A", "B"]:
        for label in ["Negative", "Mixed Negative", "Neutral Negative",
                       "Neutral Positive", "Mixed Positive", "Positive"]:
            for _ in range(10):
                rows.append({"speaker_id": spk, "sentiment_label": label})
    # Speaker B gets fewer rows for one label
    for _ in range(5):
        rows.append({"speaker_id": "C", "sentiment_label": "Negative"})
    df = pd.DataFrame(rows)
    out = filter_speakers_by_coverage(df, min_per_label=10)
    assert set(out["speaker_id"].unique()) == {"A", "B"}


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def test_rbc_perfect_positive():
    x = np.array([2.0, 3.0, 4.0])
    y = np.array([1.0, 1.0, 1.0])
    assert matched_pairs_rbc(x, y) > 0


def test_rbc_range():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    rbc = matched_pairs_rbc(x, y)
    assert -1.0 <= rbc <= 1.0


def test_concordance_prob():
    x = np.array([3.0, 3.0, 3.0])
    y = np.array([1.0, 1.0, 1.0])
    assert concordance_prob(x, y) == 1.0


def test_bh_correct_ordering():
    pvals = [0.001, 0.01, 0.5, 0.9]
    corrected = bh_correct(pvals)
    assert corrected[0] < corrected[-1]  # more significant stays lower


def test_bootstrap_ci_contains_mean():
    vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    lo, hi = bootstrap_ci(vals, n_bootstrap=500, seed=0)
    assert lo < np.mean(vals) < hi


def test_h3_check_strong():
    neg = {"kendall_p": 0.001, "linear_p": 0.001, "kendall_tau": -0.1, "linear_slope": -0.5}
    pos = {"kendall_p": 0.001, "linear_p": 0.001, "kendall_tau": 0.1, "linear_slope": 0.5}
    assert h3_check(neg, pos) == "strong"


def test_h3_check_none_same_sign():
    neg = {"kendall_p": 0.001, "linear_p": 0.001, "kendall_tau": -0.1, "linear_slope": -0.5}
    pos = {"kendall_p": 0.001, "linear_p": 0.001, "kendall_tau": -0.05, "linear_slope": -0.2}
    assert h3_check(neg, pos) == "none"


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def test_syllable_count_hr():
    assert count_syllables("politika", "HR") == 4   # po-li-ti-ka
    assert count_syllables("vlada", "HR") == 3


def test_syllable_count_pl():
    assert count_syllables("przykład", "PL") == 3   # przy-kład (ą counts)


def test_syllable_count_empty():
    assert count_syllables("", "HR") == 0
    assert count_syllables("xyz", "HR") == 0  # no vowels
