#!/usr/bin/env python3
# ============================================================
# Script:  42_plots.py
# Release: 1.0
# Version: v1.00
# Purpose: Generate all paper figures from analysis results.
#          Reads defaults from config.json.
#          For editorial control, use 42_plots_editorial.py instead.
#
# Output:  {results_dir}/figures/fig1_concordance.png
#          {results_dir}/figures/fig2_trends.png
#          {results_dir}/figures/fig3_global.png
#          {results_dir}/figures/fig4_diagnostic.png
# ============================================================

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_results_dir, get_intermediate_dir
from utils.data_utils import compute_bins, speaker_binned_means
from utils.plotting import (
    plot_concordance_bars, plot_trend_curves,
    plot_global_trend, plot_diagnostic_split, save_fig, LANG_ORDER
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--dpi", type=int, default=None)
    return p.parse_args()


def load_lang_data(langs, feats, idir, cfg):
    """Load features TSVs and compute binned means per (lang, feat)."""
    n_bins = cfg["analysis"]["n_bins"]
    binned = {}
    for lang in langs:
        path = idir / f"{lang}_features.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        df = compute_bins(df, n_bins=n_bins)
        for feat in feats:
            if feat not in df.columns:
                continue
            means = speaker_binned_means(df, [feat], n_bins=n_bins)[feat]
            binned[(lang, feat)] = means
    return binned


def main():
    args = parse_args()
    cfg = load_config(args.config)
    langs = cfg["languages"]
    feats = cfg["analysis"]["features_main"]
    idir = get_intermediate_dir(cfg)
    rdir = get_results_dir(cfg)
    fdir = rdir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    pcfg = cfg["plots"]
    dpi = args.dpi or pcfg["dpi"]
    s_min = cfg["analysis"]["sentiment_min"]
    s_max = cfg["analysis"]["sentiment_max"]

    def load_json(name):
        p = rdir / name
        return json.load(open(p)) if p.exists() else {}

    def reshape(d):
        return {tuple(k.split("_", 1)): v for k, v in d.items() if "_" in k}

    binned = load_lang_data(langs, feats, idir, cfg)
    h1 = reshape(load_json("h1_results.json"))
    h3 = reshape(load_json("h3_results.json"))
    gt_data = load_json("global_trend.json")
    split_point = gt_data.get("split_point")

    # Fig 1 — Concordance bars
    print("Fig 1: concordance bars ...")
    fig1 = plot_concordance_bars(h1, languages=langs, features=feats)
    save_fig(fig1, fdir / "fig1_concordance.png", dpi=dpi)

    # Fig 2 — Trend curves
    print("Fig 2: trend curves ...")
    fig2 = plot_trend_curves(binned, languages=langs, features=feats,
                              sentiment_min=s_min, sentiment_max=s_max)
    save_fig(fig2, fdir / "fig2_trends.png", dpi=dpi)

    # Fig 3 — Global trend
    print("Fig 3: global trend ...")
    if gt_data:
        gt_series = pd.Series(
            [np.nan if v is None else v for v in gt_data.get("values", [])],
            index=gt_data.get("bins", range(cfg["analysis"]["n_bins"]))
        )
        fig3 = plot_global_trend(gt_series, split_point=split_point,
                                  sentiment_min=s_min, sentiment_max=s_max)
        save_fig(fig3, fdir / "fig3_global.png", dpi=dpi)

    # Fig 4 — Diagnostic split
    print("Fig 4: diagnostic split ...")
    h3_reshaped = {}
    for k, v in h3.items():
        lang, feat = k
        h3_reshaped[(lang, feat)] = v
    fig4 = plot_diagnostic_split(binned, h3_reshaped, split_point or 3.5,
                                  languages=langs, features=feats,
                                  sentiment_min=s_min, sentiment_max=s_max)
    save_fig(fig4, fdir / "fig4_diagnostic.png", dpi=dpi)

    print(f"\nAll figures → {fdir}")


if __name__ == "__main__":
    main()
