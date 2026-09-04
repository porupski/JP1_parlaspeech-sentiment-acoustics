# ============================================================
# Script:  stats.py
# Release: 1.0
# Version: v1.00
# Purpose: Statistical tests — H1 Wilcoxon, H2 Kendall, BH correction, bootstrap CIs
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau, wilcoxon, ttest_1samp
from statsmodels.stats.multitest import multipletests
from typing import Optional


# ---------------------------------------------------------------------------
# Effect size helpers
# ---------------------------------------------------------------------------

def matched_pairs_rbc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Matched-pairs rank-biserial correlation for Wilcoxon signed-rank test.
    RBC = (W_pos - W_neg) / (n*(n+1)/2), range [-1, 1].
    Positive = x tends to be larger than y.
    """
    diffs = np.asarray(x) - np.asarray(y)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    W_pos = ranks[diffs > 0].sum()
    W_neg = ranks[diffs < 0].sum()
    n = len(diffs)
    return (W_pos - W_neg) / (n * (n + 1) / 2)


def concordance_prob(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) across matched pairs."""
    return float(np.mean(np.asarray(x) > np.asarray(y)))


def format_p(p: float) -> str:
    """Format p-value for display: p < 0.001 instead of p = 0.000."""
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"= {p:.3f}"
    else:
        return f"= {p:.3f}"


# ---------------------------------------------------------------------------
# H1: Wilcoxon signed-rank (neg vs pos sentiment extremes)
# ---------------------------------------------------------------------------

def h1_speaker_avg(df: pd.DataFrame,
                   feature_col: str,
                   speaker_col: str = "speaker_id",
                   label_col: str = "sentiment_label",
                   neg_label: str = "Negative",
                   pos_label: str = "Positive") -> dict:
    """
    Speaker-averaged Wilcoxon: compute mean feature per speaker per sentiment,
    then paired test across speakers.
    Returns: p, rbc, concordance, n_speakers
    """
    neg = df[df[label_col] == neg_label].groupby(speaker_col)[feature_col].mean()
    pos = df[df[label_col] == pos_label].groupby(speaker_col)[feature_col].mean()
    common = neg.index.intersection(pos.index)
    neg, pos = neg[common].values, pos[common].values
    if len(neg) < 5:
        return {"p": np.nan, "rbc": np.nan, "concordance": np.nan, "n": len(neg)}
    _, p = wilcoxon(neg, pos)
    rbc = matched_pairs_rbc(neg, pos)
    conc = concordance_prob(neg, pos)
    return {"p": p, "rbc": rbc, "concordance": conc, "n": len(neg)}


def h1_utterance_level(df: pd.DataFrame,
                        feature_col: str,
                        n_pairs: int = 10,
                        seed: int = 42,
                        speaker_col: str = "speaker_id",
                        label_col: str = "sentiment_label",
                        neg_label: str = "Negative",
                        pos_label: str = "Positive") -> dict:
    """
    Utterance-level Wilcoxon: randomly sample n_pairs neg/pos pairs per speaker,
    pool across speakers, run single test. Seed ensures reproducibility.
    Returns: p, rbc, concordance, n_pairs_total
    """
    rng = np.random.default_rng(seed)
    neg_vals, pos_vals = [], []
    for spk, grp in df.groupby(speaker_col):
        neg_utt = grp[grp[label_col] == neg_label][feature_col].dropna().values
        pos_utt = grp[grp[label_col] == pos_label][feature_col].dropna().values
        if len(neg_utt) < 1 or len(pos_utt) < 1:
            continue
        k = min(n_pairs, len(neg_utt), len(pos_utt))
        neg_vals.extend(rng.choice(neg_utt, k, replace=False))
        pos_vals.extend(rng.choice(pos_utt, k, replace=False))
    neg_vals, pos_vals = np.array(neg_vals), np.array(pos_vals)
    if len(neg_vals) < 5:
        return {"p": np.nan, "rbc": np.nan, "concordance": np.nan, "n": len(neg_vals)}
    _, p = wilcoxon(neg_vals, pos_vals)
    rbc = matched_pairs_rbc(neg_vals, pos_vals)
    conc = concordance_prob(neg_vals, pos_vals)
    return {"p": p, "rbc": rbc, "concordance": conc, "n": len(neg_vals)}


# ---------------------------------------------------------------------------
# H2: Kendall's tau across sentiment continuum (per speaker → one-sample t-test)
# ---------------------------------------------------------------------------

def h2_kendall(df: pd.DataFrame,
               feature_col: str,
               use_bins: bool = True,
               n_bins: int = 60,
               speaker_col: str = "speaker_id",
               bin_col: str = "bin",
               score_col: str = "sentiment_score",
               sig_alpha: float = 0.05) -> dict:
    """
    Kendall's tau per speaker, then one-sample t-test on mean tau.

    use_bins=True (default, matches paper caption):
        Compute per-speaker bin means across n_bins bins, then tau over bin indices.
    use_bins=False (investigate §7.1 bug):
        Compute tau over raw utterance scores per speaker.

    Returns: p (t-test), mean_tau, n_speakers, n_sig_speakers
    """
    taus = []
    sig_count = 0
    for spk, grp in df.groupby(speaker_col):
        grp = grp.dropna(subset=[feature_col])
        if use_bins:
            curve = grp.groupby(bin_col)[feature_col].mean().dropna()
            if len(curve) < 5:
                continue
            x, y = curve.index.values, curve.values
        else:
            if len(grp) < 5:
                continue
            x = grp[score_col].values
            y = grp[feature_col].values
        tau, p_spk = kendalltau(x, y)
        taus.append(tau)
        if p_spk < sig_alpha:
            sig_count += 1
    if len(taus) < 3:
        return {"p": np.nan, "mean_tau": np.nan, "n_speakers": len(taus), "n_sig_speakers": sig_count}
    t_stat, p = ttest_1samp(taus, 0.0)
    return {
        "p": p,
        "mean_tau": float(np.mean(taus)),
        "std_tau": float(np.std(taus)),
        "n_speakers": len(taus),
        "n_sig_speakers": sig_count,
    }


# ---------------------------------------------------------------------------
# H3: Split-side Kendall + linear regression
# ---------------------------------------------------------------------------

def h3_split_side(binned_means: pd.Series,
                   split_bin: int,
                   side: str = "negative") -> dict:
    """
    Run Kendall tau + linear regression on one side of the split.
    side: 'negative' (bins 0..split_bin) or 'positive' (bins split_bin..end)
    binned_means: Series indexed by bin number.
    Returns: {kendall_p, kendall_tau, linear_p, linear_slope}
    """
    if side == "negative":
        segment = binned_means.loc[:split_bin].dropna()
    else:
        segment = binned_means.loc[split_bin:].dropna()
    if len(segment) < 3:
        return {"kendall_p": np.nan, "kendall_tau": np.nan,
                "linear_p": np.nan, "linear_slope": np.nan}
    x = segment.index.values.astype(float)
    y = segment.values
    tau, kp = kendalltau(x, y)
    slope, intercept, r, lp, se = stats.linregress(x, y)
    return {"kendall_p": kp, "kendall_tau": tau, "linear_p": lp, "linear_slope": slope}


def h3_check(neg_result: dict, pos_result: dict) -> str:
    """
    Determine arousal support level from split-side results.
    Returns: 'strong' (both sides sig, opposite slopes), 'partial' (one side + opposite slope), 'none'
    """
    def sig(r, key):
        p = r.get(key, np.nan)
        return not np.isnan(p) and p < 0.05

    neg_k_sig = sig(neg_result, "kendall_p")
    neg_l_sig = sig(neg_result, "linear_p")
    pos_k_sig = sig(pos_result, "kendall_p")
    pos_l_sig = sig(pos_result, "linear_p")

    neg_slope = neg_result.get("linear_slope", np.nan)
    pos_slope = pos_result.get("linear_slope", np.nan)
    opposite = (not np.isnan(neg_slope) and not np.isnan(pos_slope)
                and neg_slope * pos_slope < 0)  # opposite signs

    if not opposite:
        return "none"
    neg_supported = neg_k_sig or neg_l_sig
    pos_supported = pos_k_sig or pos_l_sig
    if neg_supported and pos_k_sig and pos_l_sig:
        return "strong"
    elif neg_supported and pos_supported:
        return "partial"
    return "none"


# ---------------------------------------------------------------------------
# Multiple comparison correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

def bh_correct(pvalues: list[float], alpha: float = 0.05) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction.
    Returns array of corrected p-values (same order as input).
    NaN values are preserved in-place; correction applied to valid p-values only.
    """
    pvals = np.array(pvalues, dtype=float)
    valid_mask = ~np.isnan(pvals)
    corrected = pvals.copy()
    if valid_mask.sum() > 0:
        _, pvals_corr, _, _ = multipletests(
            pvals[valid_mask], alpha=alpha, method="fdr_bh"
        )
        corrected[valid_mask] = pvals_corr
    return corrected


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray,
                 n_bootstrap: int = 1000,
                 ci_level: float = 0.95,
                 seed: int = 42) -> tuple[float, float]:
    """
    Bootstrap CI on the mean of values.
    Returns (lower, upper) percentile CI.
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci_level
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
