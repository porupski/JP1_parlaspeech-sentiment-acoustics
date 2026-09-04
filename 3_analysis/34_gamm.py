#!/usr/bin/env python3
# ============================================================
# Script:  34_gamm.py
# Release: 1.0
# Version: v1.00
# Purpose: GAMM analysis via rpy2 (optional, booleaned off by default).
#          Enable: set "enable_gamm": true in config.json.
#
# Input:   {intermediate_dir}/{lang}_features.tsv
# Output:  {results_dir}/gamm_results.json
# ============================================================

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir, get_results_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    return p.parse_args()


def run_gamm_r(df: pd.DataFrame, feature: str, cfg: dict) -> dict:
    """Run GAMM via rpy2. Returns dict of model summary stats."""
    import os
    os.environ["R_HOME"] = cfg["paths"]["r_home"]
    os.environ["R_LIBS_USER"] = cfg["paths"]["r_libs"]

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    pandas2ri.activate()
    mgcv = importr("mgcv")

    r_df = pandas2ri.py2rpy(df[["sentiment_score", feature, "speaker_id"]].dropna())
    formula = ro.Formula(f"{feature} ~ s(sentiment_score) + s(speaker_id, bs='re')")
    model = mgcv.gam(formula, data=r_df, method="REML")
    summary = ro.r["summary"](model)

    # Extract smooth term edf and p-value
    edf = float(summary.rx2("edf")[0])
    s_table = summary.rx2("s.table")
    p_smooth = float(s_table[3])  # p for s(sentiment_score)

    return {"edf": edf, "p_smooth": p_smooth}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if not cfg["analysis"]["enable_gamm"]:
        print("GAMM disabled in config (enable_gamm = false). Exiting.")
        print("To enable: set 'enable_gamm': true in config.json")
        return

    langs = args.langs or cfg["languages"]
    idir = get_intermediate_dir(cfg)
    rdir = get_results_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    features = cfg["analysis"]["features_main"]

    all_results = {}
    for lang in langs:
        path = idir / f"{lang}_features.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        print(f"\n[{lang}]")
        for feat in features:
            if feat not in df.columns:
                continue
            print(f"  Running GAMM for {feat} ...")
            result = run_gamm_r(df, feat, cfg)
            all_results[f"{lang}_{feat}"] = result
            print(f"    edf={result['edf']:.2f}  p={result['p_smooth']:.4f}")

    out = rdir / "gamm_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWritten → {out}")


if __name__ == "__main__":
    main()
