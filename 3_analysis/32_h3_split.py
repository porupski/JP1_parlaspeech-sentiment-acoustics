#!/usr/bin/env python3
# ============================================================
# Script:  32_h3_split.py
# Release: 1.0
# Version: v1.00
# Purpose: H3 — Inflection-based split analysis.
#          1. Build global trend (min-max normalised, averaged across speakers/languages)
#          2. Detect split point from global minimum (or use config override)
#          3. Per-side Kendall + linear regression for each lang×feature
#
# Input:   {intermediate_dir}/{lang}_features.tsv (all languages)
# Output:  {results_dir}/h3_results.json
#          {results_dir}/global_trend.json  (bin means + split point)
# ============================================================

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kendalltau, ttest_1samp, linregress

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir, get_results_dir
from utils.data_utils import compute_bins
from utils.stats import h3_check, bh_correct


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    return p.parse_args()


def build_global_trend(lang_curves: dict, features: list[str],
                        n_bins: int, weighting: str = "equal_language"
                        ) -> "tuple[pd.Series, pd.Series]":
    """
    Build global trend curve (both weighting schemes in one pass).

    Returns (equal_language_series, weighted_speaker_series), both indexed 0..n_bins-1.

    equal_language:   each language contributes equally (mean of per-language means)
    weighted_speaker: pool all normalised speaker curves across languages before averaging
                      (larger languages contribute proportionally more)

    lang_curves: {lang: {spk_id: DataFrame(index=bin, cols=features)}}
    """
    lang_trends = {}
    all_spk_curves: list[np.ndarray] = []   # for weighted_speaker

    for lang, spk_curves in lang_curves.items():
        all_normalised = []
        for feat in features:
            spk_normalised = []
            for spk_id, curve_df in spk_curves.items():
                if feat not in curve_df.columns:
                    continue
                y = curve_df[feat].reindex(range(n_bins)).values.astype(float)
                valid = ~np.isnan(y)
                if valid.sum() < 5:
                    continue
                ymin, ymax = np.nanmin(y), np.nanmax(y)
                if ymax == ymin:
                    continue
                y_norm = (y - ymin) / (ymax - ymin)
                spk_normalised.append(y_norm)
                all_spk_curves.append(y_norm)   # pool for weighted_speaker
            if spk_normalised:
                all_normalised.append(np.nanmean(spk_normalised, axis=0))
        if all_normalised:
            lang_trends[lang] = np.nanmean(all_normalised, axis=0)

    nan_series = pd.Series(np.nan, index=range(n_bins))

    if not lang_trends:
        return nan_series, nan_series

    eq_mean = np.nanmean(list(lang_trends.values()), axis=0)
    wt_mean = np.nanmean(all_spk_curves, axis=0) if all_spk_curves else eq_mean

    return pd.Series(eq_mean, index=range(n_bins)), pd.Series(wt_mean, index=range(n_bins))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    langs = args.langs or cfg["languages"]
    idir = get_intermediate_dir(cfg)
    rdir = get_results_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)

    features = cfg["analysis"]["features_main"] + cfg["analysis"]["features_appendix"]
    n_bins = cfg["analysis"]["n_bins"]
    s_min = cfg["analysis"]["sentiment_min"]
    s_max = cfg["analysis"]["sentiment_max"]
    weighting = cfg["analysis"]["global_trend_weighting"]
    override_split = cfg["analysis"].get("split_point")

    # Load all language data and build per-speaker curves
    lang_df = {}
    lang_curves = {}  # {lang: {spk: DataFrame(bin × feats)}}
    for lang in langs:
        path = idir / f"{lang}_features.tsv"
        if not path.exists():
            print(f"[{lang}] {path} not found. Skipping.")
            continue
        df = pd.read_csv(path, sep="\t")
        df = compute_bins(df, n_bins=n_bins)
        lang_df[lang] = df

        # per-speaker curves for each feature
        spk_curves = {}
        for spk, grp in df.groupby("speaker_id"):
            curve_row = {}
            for feat in features:
                if feat not in grp.columns:
                    continue
                bin_means = grp.dropna(subset=[feat]).groupby("bin")[feat].mean()
                for b in range(n_bins):
                    curve_row.setdefault(b, {})[feat] = bin_means.get(b, np.nan)
            spk_curves[spk] = pd.DataFrame(curve_row).T
        lang_curves[lang] = spk_curves

    # Build global trend — both weighting schemes in one pass
    print(f"\nBuilding global trend (primary: {weighting}) ...")
    gt_equal, gt_weighted = build_global_trend(lang_curves, features, n_bins, weighting)
    global_trend = gt_equal if weighting == "equal_language" else gt_weighted

    # Find split point (always from the primary weighting curve)
    if override_split is not None:
        split_point = float(override_split)
        print(f"Using config override split point: {split_point}")
    else:
        split_bin = int(np.nanargmin(global_trend.values))
        split_point = s_min + (split_bin + 0.5) * (s_max - s_min) / n_bins
        print(f"Detected split point: {split_point:.3f} (bin {split_bin})")
        # Also report weighted_speaker minimum for comparison
        wt_min_bin = int(np.nanargmin(gt_weighted.values))
        wt_split = s_min + (wt_min_bin + 0.5) * (s_max - s_min) / n_bins
        print(f"  weighted_speaker minimum:  {wt_split:.3f} (bin {wt_min_bin})")

    split_bin = int((split_point - s_min) / (s_max - s_min) * n_bins)
    split_bin = max(5, min(split_bin, n_bins - 6))  # guard against edge bins

    def _series_to_list(s):
        return [None if np.isnan(v) else float(v) for v in s.values]

    # Save global trend (both curves)
    global_out = rdir / "global_trend.json"
    with open(global_out, "w") as f:
        json.dump({
            "bins": list(range(n_bins)),
            "values": _series_to_list(gt_equal),
            "values_weighted_speaker": _series_to_list(gt_weighted),
            "split_bin": split_bin,
            "split_point": split_point,
            "weighting": weighting,
        }, f, indent=2)
    print(f"Written → {global_out}")

    # Pre-compute bin-centre sentiment values for linear regression x-axis
    bin_centers = pd.Series(
        s_min + (np.arange(n_bins) + 0.5) * (s_max - s_min) / n_bins,
        index=range(n_bins),
    )

    def _per_speaker_kendall(df_feat: pd.DataFrame, feat: str,
                              lo: float, hi: float) -> dict:
        """Per-speaker Kendall on raw utterances in [lo, hi), then t-test on mean tau."""
        taus, sig_count = [], 0
        mask = (df_feat["sentiment_score"] >= lo) & (df_feat["sentiment_score"] < hi)
        side_df = df_feat[mask]
        for _, grp in side_df.groupby("speaker_id"):
            if len(grp) < 3:
                continue
            tau, p = kendalltau(grp["sentiment_score"].values, grp[feat].values)
            taus.append(tau)
            if p < 0.05:
                sig_count += 1
        if len(taus) < 3:
            return {"kendall_p": np.nan, "kendall_tau": np.nan,
                    "n_speakers": len(taus), "n_sig_speakers": sig_count}
        _, p = ttest_1samp(taus, 0.0)
        return {"kendall_p": float(p), "kendall_tau": float(np.mean(taus)),
                "std_tau": float(np.std(taus)),
                "n_speakers": len(taus), "n_sig_speakers": sig_count}

    def _linreg_side(binned: pd.Series, side: str) -> dict:
        """Linear regression on population bin means; x in sentiment units."""
        seg = binned.loc[:split_bin] if side == "negative" else binned.loc[split_bin:]
        seg = seg.dropna()
        if len(seg) < 3:
            return {"linear_p": np.nan, "linear_slope": np.nan}
        x = bin_centers.loc[seg.index].values
        slope, _, _, lp, _ = linregress(x, seg.values)
        return {"linear_p": float(lp), "linear_slope": float(slope)}

    # Per-language, per-feature split analysis
    all_results = {}
    all_pvals_k = []
    all_keys = []

    for lang, df in lang_df.items():
        print(f"\n[{lang}]")
        for feat in features:
            if feat not in df.columns:
                continue
            df_feat = df.dropna(subset=[feat])

            # Per-speaker Kendall on raw utterances, each side of split
            neg_k = _per_speaker_kendall(df_feat, feat, lo=s_min, hi=split_point)
            pos_k = _per_speaker_kendall(df_feat, feat, lo=split_point, hi=s_max + 1e-9)

            # Linear regression on population binned means (x = sentiment value)
            binned = df_feat.groupby("bin")[feat].mean().reindex(range(n_bins))
            neg_lin = _linreg_side(binned, "negative")
            pos_lin = _linreg_side(binned, "positive")

            neg_r = {**neg_k, **neg_lin}
            pos_r = {**pos_k, **pos_lin}
            check = h3_check(neg_r, pos_r)

            key = f"{lang}_{feat}"
            all_results[key] = {"neg": neg_r, "pos": pos_r, "check": check,
                                  "split_bin": split_bin, "split_point": split_point}

            tau_neg = neg_r.get("kendall_tau") or float("nan")
            tau_pos = pos_r.get("kendall_tau") or float("nan")
            sym = {"strong": "✓", "partial": "+", "none": "×"}[check]
            print(f"  {feat:25s}  "
                  f"neg: τ={tau_neg:.3f} (n={neg_r.get('n_speakers','?')}) | "
                  f"pos: τ={tau_pos:.3f} (n={pos_r.get('n_speakers','?')}) | {sym}")

            all_pvals_k.append(neg_r.get("kendall_p", np.nan))
            all_pvals_k.append(pos_r.get("kendall_p", np.nan))
            all_keys.append((key, "neg"))
            all_keys.append((key, "pos"))

    # BH correction on all Kendall p-values in H3
    corr = bh_correct(all_pvals_k, alpha=cfg["analysis"]["bh_alpha"])
    for (key, side), p_corr in zip(all_keys, corr):
        all_results[key][side]["kendall_p_bh"] = float(p_corr)

    out = rdir / "h3_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\nWritten → {out}")


if __name__ == "__main__":
    main()
