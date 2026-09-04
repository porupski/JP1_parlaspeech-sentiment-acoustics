#!/usr/bin/env python3
# ============================================================
# Script:  31_h2_monotonic.py
# Release: 1.0
# Version: v1.00
# Purpose: H2 — Kendall's tau across the full sentiment continuum.
#          Per speaker, then one-sample t-test on mean tau.
#          Bootstrap CIs on mean tau.
#
# NOTE §7.1: The paper caption says tau is computed per speaker on 60 bins.
#    But tau values and sig% are inconsistent with n=60.
#    Config flag 'kendall_on_bins' (true=bins, false=raw utterances).
#    Run both and compare — if results differ materially, document and ask.
#
# Input:   {intermediate_dir}/{lang}_features.tsv
# Output:  {results_dir}/h2_results.json
# ============================================================

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir, get_results_dir
from utils.data_utils import compute_bins
from utils.stats import h2_kendall, bh_correct, bootstrap_ci


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    langs = args.langs or cfg["languages"]
    idir = get_intermediate_dir(cfg)
    rdir = get_results_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)

    features = cfg["analysis"]["features_main"] + cfg["analysis"]["features_appendix"]
    n_bins = cfg["analysis"]["n_bins"]
    use_bins = cfg["analysis"]["kendall_on_bins"]
    seed = cfg["seed"]
    bootstrap_n = cfg["analysis"]["bootstrap_n"]
    ci_level = cfg["analysis"]["bootstrap_ci_level"]

    print(f"Kendall mode: {'60 bins per speaker' if use_bins else 'raw utterances per speaker'}")
    print("(§7.1: if sig% is inconsistent with n=60, switch kendall_on_bins to false)")

    all_results = {}

    for lang in langs:
        path = idir / f"{lang}_features.tsv"
        if not path.exists():
            print(f"[{lang}] {path} not found. Skipping.")
            continue
        df = pd.read_csv(path, sep="\t")
        df = compute_bins(df, n_bins=n_bins)
        print(f"\n[{lang}] {len(df):,} utterances")

        for feat in features:
            if feat not in df.columns:
                continue
            df_feat = df.dropna(subset=[feat])

            result = h2_kendall(df_feat, feat, use_bins=use_bins, n_bins=n_bins)

            # Bootstrap CI on mean tau
            # Collect per-speaker taus for CI
            from scipy.stats import kendalltau
            spk_taus = []
            for spk, grp in df_feat.groupby("speaker_id"):
                grp = grp.dropna(subset=[feat])
                if use_bins:
                    curve = grp.groupby("bin")[feat].mean().dropna()
                    if len(curve) < 5:
                        continue
                    x, y = curve.index.values, curve.values
                else:
                    if len(grp) < 5:
                        continue
                    x = grp["sentiment_score"].values
                    y = grp[feat].values
                tau, _ = kendalltau(x, y)
                if not np.isnan(tau):
                    spk_taus.append(tau)

            ci_lo, ci_hi = bootstrap_ci(np.array(spk_taus), n_bootstrap=bootstrap_n,
                                         ci_level=ci_level, seed=seed)
            result["ci_lo"] = ci_lo
            result["ci_hi"] = ci_hi

            key = f"{lang}_{feat}"
            all_results[key] = result

            print(f"  {feat:25s}  p={result['p']:.4f} "
                  f"τ={result['mean_tau']:.4f} "
                  f"[{ci_lo:.4f}, {ci_hi:.4f}] "
                  f"sig_spk={result['n_sig_speakers']}/{result['n_speakers']}")

    # BH correction
    keys = list(all_results.keys())
    pvals = [all_results[k]["p"] for k in keys]
    corr_pvals = bh_correct(pvals, alpha=cfg["analysis"]["bh_alpha"])
    for k, p_corr in zip(keys, corr_pvals):
        all_results[k]["p_bh"] = float(p_corr)

    out = rdir / "h2_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\nWritten → {out}")


if __name__ == "__main__":
    main()
