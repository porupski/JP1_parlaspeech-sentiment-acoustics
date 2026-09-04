#!/usr/bin/env python3
# ============================================================
# Script:  23_normalize.py
# Release: 1.0
# Version: v1.00
# Purpose: Add per-session and per-speaker normalized feature columns
#          to the combined feature TSV.
#
# Input:   {intermediate_dir}/{lang}_features.tsv  (from 11_join.py)
# Output:  {intermediate_dir}/{lang}_features.tsv  (overwritten, adds *_norm cols)
#
# Added columns:
#   intensity_norm   — per-session (session_id) z-score of intensity_raw
#   f0_norm          — per-speaker z-score of f0_raw
#                      (pitch less sensitive to recording conditions,
#                       but cheap to add; kept as separate track)
# ============================================================

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir
from utils.data_utils import write_tsv


def zscore_within_group(series: pd.Series, groups: pd.Series) -> pd.Series:
    """Z-score series within each group. Groups with std=0 → NaN."""
    result = series.copy().astype(float)
    for grp_val, idx in series.groupby(groups).groups.items():
        vals = series.loc[idx].dropna()
        if len(vals) < 2:
            result.loc[idx] = np.nan
            continue
        mu, sigma = vals.mean(), vals.std()
        if sigma == 0:
            result.loc[idx] = np.nan
        else:
            result.loc[idx] = (series.loc[idx] - mu) / sigma
    return result


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

    for lang in langs:
        path = idir / f"{lang}_features.tsv"
        if not path.exists():
            print(f"[{lang}] {path} not found. Run 11_join.py first.")
            continue

        df = pd.read_csv(path, sep="\t")
        print(f"\n[{lang}] Adding normalized columns to {len(df):,} rows ...")

        # Intensity: z-score within session_id
        if "intensity_raw" in df.columns and "session_id" in df.columns:
            df["intensity_norm"] = zscore_within_group(
                df["intensity_raw"], df["session_id"]
            )
            n_valid = df["intensity_norm"].notna().sum()
            print(f"  intensity_norm: {n_valid:,}/{len(df):,} valid")
        else:
            print("  [WARN] intensity_raw or session_id missing — skipping intensity_norm")

        # F0: z-score within speaker_id
        if "f0_raw" in df.columns and "speaker_id" in df.columns:
            df["f0_norm"] = zscore_within_group(
                df["f0_raw"], df["speaker_id"]
            )
            n_valid = df["f0_norm"].notna().sum()
            print(f"  f0_norm: {n_valid:,}/{len(df):,} valid")
        else:
            print("  [WARN] f0_raw or speaker_id missing — skipping f0_norm")

        write_tsv(df, path)
        print(f"  Updated → {path}")


if __name__ == "__main__":
    main()
