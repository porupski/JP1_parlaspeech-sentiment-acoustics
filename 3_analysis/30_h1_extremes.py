#!/usr/bin/env python3
# ============================================================
# Script:  30_h1_extremes.py
# Release: 1.0
# Version: v1.00
# Purpose: H1 — Wilcoxon signed-rank comparing Negative vs Positive
#          sentiment extremes. Speaker-averaged and utterance-level.
#
# Input:   {intermediate_dir}/{lang}_features.tsv (one per language)
# Output:  {results_dir}/h1_results.json
#          Keyed by (lang, feature): {speaker_avg: {...}, utterance_level: {...}}
# ============================================================

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir, get_results_dir
from utils.stats import h1_speaker_avg, h1_utterance_level, bh_correct


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

    features = (cfg["analysis"]["features_main"]
                + cfg["analysis"]["features_appendix"])
    seed = cfg["seed"]
    n_pairs = cfg["filter"]["n_pairs_h1"]

    all_results = {}

    for lang in langs:
        path = idir / f"{lang}_features.tsv"
        if not path.exists():
            print(f"[{lang}] {path} not found. Skipping.")
            continue
        df = pd.read_csv(path, sep="\t")
        print(f"\n[{lang}] {len(df):,} utterances, "
              f"{df['speaker_id'].nunique()} speakers")

        for feat in features:
            if feat not in df.columns:
                continue
            df_feat = df.dropna(subset=[feat])

            sa = h1_speaker_avg(df_feat, feat)
            ul = h1_utterance_level(df_feat, feat, n_pairs=n_pairs, seed=seed)

            key = f"{lang}_{feat}"
            all_results[key] = {"speaker_avg": sa, "utterance_level": ul}

            print(f"  {feat:25s}  SA: p={sa['p']:.4f} RBC={sa['rbc']:.3f} "
                  f"P(N>P)={sa['concordance']:.3f} | "
                  f"UL: p={ul['p']:.4f} RBC={ul['rbc']:.3f}")

    # BH correction across all p-values (speaker-avg track)
    keys = list(all_results.keys())
    sa_pvals = [all_results[k]["speaker_avg"]["p"] for k in keys]
    sa_corr = bh_correct(sa_pvals, alpha=cfg["analysis"]["bh_alpha"])
    for k, p_corr in zip(keys, sa_corr):
        all_results[k]["speaker_avg"]["p_bh"] = float(p_corr)

    ul_pvals = [all_results[k]["utterance_level"]["p"] for k in keys]
    ul_corr = bh_correct(ul_pvals, alpha=cfg["analysis"]["bh_alpha"])
    for k, p_corr in zip(keys, ul_corr):
        all_results[k]["utterance_level"]["p_bh"] = float(p_corr)

    out = rdir / "h1_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\nWritten → {out}")


if __name__ == "__main__":
    main()
