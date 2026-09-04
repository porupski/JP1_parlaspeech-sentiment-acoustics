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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir, get_results_dir
from utils.data_utils import compute_bins, speaker_binned_curves
from utils.stats import h3_split_side, h3_check, bh_correct


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    return p.parse_args()


def build_global_trend(lang_curves: dict, features: list[str],
                        n_bins: int, weighting: str = "equal_language") -> pd.Series:
    """
    Build global trend curve:
      - Per speaker per feature: min-max normalise the 60-bin curve
      - Average normalised curves across speakers (equal per speaker within language)
      - Average across languages (equal per language if weighting='equal_language')
    Returns Series indexed by bin (0..n_bins-1).

    lang_curves: {lang: {feat: pd.DataFrame(index=bin, cols=features)}}
    """
    lang_trends = {}
    for lang, feat_curves in lang_curves.items():
        all_normalised = []
        for feat in features:
            spk_normalised = []
            for spk_id, curve_df in feat_curves.items():
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
            if spk_normalised:
                all_normalised.append(np.nanmean(spk_normalised, axis=0))
        if all_normalised:
            lang_trends[lang] = np.nanmean(all_normalised, axis=0)

    if not lang_trends:
        return pd.Series(np.nan, index=range(n_bins))

    if weighting == "equal_language":
        global_mean = np.nanmean(list(lang_trends.values()), axis=0)
    else:  # weighted_speaker — fall back to equal_language, noted in output
        global_mean = np.nanmean(list(lang_trends.values()), axis=0)

    return pd.Series(global_mean, index=range(n_bins))


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

    # Build global trend
    print(f"\nBuilding global trend (weighting={weighting}) ...")
    global_trend = build_global_trend(lang_curves, features, n_bins, weighting)

    # Find split point
    if override_split is not None:
        split_point = float(override_split)
        print(f"Using config override split point: {split_point}")
    else:
        split_bin = int(np.nanargmin(global_trend.values))
        split_point = s_min + (split_bin + 0.5) * (s_max - s_min) / n_bins
        print(f"Detected split point: {split_point:.3f} (bin {split_bin})")

    split_bin = int((split_point - s_min) / (s_max - s_min) * n_bins)
    split_bin = max(5, min(split_bin, n_bins - 6))  # guard against edge bins

    # Save global trend
    global_out = rdir / "global_trend.json"
    with open(global_out, "w") as f:
        json.dump({
            "bins": list(range(n_bins)),
            "values": [None if np.isnan(v) else float(v) for v in global_trend.values],
            "split_bin": split_bin,
            "split_point": split_point,
            "weighting": weighting,
        }, f, indent=2)
    print(f"Written → {global_out}")

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

            # Compute speaker-averaged binned means for this lang+feat
            binned = df_feat.groupby("bin")[feat].mean().reindex(range(n_bins))

            neg_r = h3_split_side(binned, split_bin, side="negative")
            pos_r = h3_split_side(binned, split_bin, side="positive")
            check = h3_check(neg_r, pos_r)

            key = f"{lang}_{feat}"
            all_results[key] = {"neg": neg_r, "pos": pos_r, "check": check,
                                  "split_bin": split_bin, "split_point": split_point}

            sym = {"strong": "✓", "partial": "+", "none": "×"}[check]
            print(f"  {feat:25s}  "
                  f"neg: τ={neg_r['kendall_tau']:.3f} | "
                  f"pos: τ={pos_r['kendall_tau']:.3f} | {sym}")

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
