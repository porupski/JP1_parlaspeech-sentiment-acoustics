#!/usr/bin/env python3
# ============================================================
# Script:  34_h3_quadratic.py
# Release: 1.0
# Version: v1.00
# Purpose: H3 — Arousal Hypothesis via quadratic curvature.
#
# For each language × feature, fit  y = β₀ + β₁·s + β₂·s²  on the
# 60-bin speaker-averaged curve. β₂ is the arousal-like curvature term.
#
# AH decision rule (per cell):
#     arousal-positive if p(β₂) < 0.05 AND β₂ > 0 AND vertex ∈ (0, 5)
#     inverted-U       if p(β₂) < 0.05 AND β₂ < 0 AND vertex ∈ (0, 5)
#     linear           otherwise (including out-of-range vertex)
#
# Hard vertex boundary: (0, 5) exclusive. A cell with a significant β₂ but
# a vertex outside this range is treated as monotonic-within-scale — the
# curvature is real but not observed in the informative sentiment range.
#
# Replaces 32_h3_split.py as the primary AH test. Which script runs is
# controlled by config `analysis.ah_method: "quadratic" | "split"`
# (default: "quadratic"). Both scripts write to `results/h3_results.json`
# so downstream table/plot code continues to read one canonical file.
# Additionally, this script always writes `h3_quadratic_results.json`
# so both AH representations can co-exist on disk.
#
# Input:   {intermediate_dir}/{lang}_features.tsv
# Output:  {results_dir}/h3_results.json               (canonical active file)
#          {results_dir}/h3_quadratic_results.json     (always written)
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
from utils.data_utils import compute_bins
from utils.stats import bh_correct


VERTEX_LO = 0.0   # hard lower bound (exclusive)
VERTEX_HI = 5.0   # hard upper bound (exclusive)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    return p.parse_args()


def ols_quadratic(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = β₀ + β₁ x + β₂ x². Return coefficients, p-values, R², vertex,
    shape label. Manual OLS: β = (XᵀX)⁻¹ Xᵀy; two-sided t-test on β₂.

    Vertex boundary is hard: p(β₂) significant but vertex ∉ (0, 5) → shape="linear".
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(y)
    empty = {k: None for k in
             ("n", "b0", "b1", "b2", "p_b1", "p_b2", "r2_lin",
              "r2_quad", "vertex_s", "shape")}
    if n < 5:
        return empty
    X = np.column_stack([np.ones(n), x, x * x])
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return empty
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    sigma2 = float(resid @ resid) / (n - 3)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    tvals = beta / se
    pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), df=n - 3))

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2_quad = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    Xl = np.column_stack([np.ones(n), x])
    bl = np.linalg.inv(Xl.T @ Xl) @ Xl.T @ y
    r2_lin = 1.0 - float(((y - Xl @ bl) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan

    b1, b2, p_b2 = float(beta[1]), float(beta[2]), float(pvals[2])
    vertex_s = (-b1 / (2 * b2)) if abs(b2) > 1e-12 else float("inf")

    in_range = VERTEX_LO < vertex_s < VERTEX_HI
    if p_b2 < 0.05 and in_range and b2 > 0:
        shape = "U"
    elif p_b2 < 0.05 and in_range and b2 < 0:
        shape = "inverted_U"
    else:
        shape = "linear"

    return {
        "n":        int(n),
        "b0":       float(beta[0]),
        "b1":       b1,
        "b2":       b2,
        "p_b1":     float(pvals[1]),
        "p_b2":     p_b2,
        "r2_lin":   float(r2_lin),
        "r2_quad":  float(r2_quad),
        "vertex_s": float(vertex_s),
        "shape":    shape,
    }


def speaker_averaged_curve(df: pd.DataFrame, feat: str, n_bins: int) -> np.ndarray:
    """Mean across speakers of each speaker's per-bin mean."""
    if feat not in df.columns:
        return np.full(n_bins, np.nan)
    curves = []
    for _spk, grp in df.dropna(subset=[feat]).groupby("speaker_id"):
        c = grp.groupby("bin")[feat].mean().reindex(range(n_bins)).values
        curves.append(c)
    if not curves:
        return np.full(n_bins, np.nan)
    return np.nanmean(curves, axis=0)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    langs = args.langs or cfg["languages"]
    idir = get_intermediate_dir(cfg)
    rdir = get_results_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)

    features = cfg["analysis"]["features_main"] + cfg["analysis"]["features_appendix"]
    n_bins   = cfg["analysis"]["n_bins"]
    s_min    = cfg["analysis"]["sentiment_min"]
    s_max    = cfg["analysis"]["sentiment_max"]
    x = 0.5 * (np.linspace(s_min, s_max, n_bins + 1)[:-1] +
                np.linspace(s_min, s_max, n_bins + 1)[1:])

    print(f"H3-quadratic  ·  vertex boundary = ({VERTEX_LO}, {VERTEX_HI}) exclusive")
    print(f"features: {features}")

    all_results: dict = {}
    all_p_b2: list = []
    all_keys: list = []

    for lang in langs:
        path = idir / f"{lang}_features.tsv"
        if not path.exists():
            print(f"[{lang}] {path} not found. Skipping."); continue
        df = pd.read_csv(path, sep="\t")
        df = compute_bins(df, n_bins=n_bins)
        print(f"\n[{lang}] {len(df):,} utterances")

        for feat in features:
            if feat not in df.columns:
                continue
            y = speaker_averaged_curve(df, feat, n_bins)
            r = ols_quadratic(x, y)
            key = f"{lang}_{feat}"
            all_results[key] = r
            if r["p_b2"] is not None:
                all_p_b2.append(r["p_b2"])
                all_keys.append(key)
            v_disp = f"{r['vertex_s']:+.2f}" if (r['vertex_s'] is not None and
                                                  np.isfinite(r['vertex_s']))\
                                             else "  n/a"
            print(f"  {feat:22s}  β₂={r['b2']:+.4g}  p(β₂)="
                  f"{'<1e-4' if (r['p_b2'] is not None and r['p_b2']<1e-4) else f'{r['p_b2']:.4f}'}  "
                  f"vertex={v_disp}  {r['shape']}")

    # BH correction on p(β₂) across all cells
    if all_p_b2:
        corr = bh_correct(all_p_b2, alpha=cfg["analysis"]["bh_alpha"])
        for key, p_corr in zip(all_keys, corr):
            all_results[key]["p_b2_bh"] = float(p_corr)

    # Summary counts by shape
    counts = {"U": 0, "inverted_U": 0, "linear": 0}
    for r in all_results.values():
        if r.get("shape") in counts:
            counts[r["shape"]] += 1
    print(f"\nShape totals across all lang × feature cells:")
    for k, v in counts.items():
        print(f"  {k:12s}  {v}")

    # Header metadata at top level (underscore-prefixed so callers that iterate
    # cells can filter it out). Cells stored flat alongside — matches 32's
    # layout so downstream code that iterates {lang_feat: {...}} keeps working.
    payload = {
        "_method":          "quadratic",
        "_vertex_boundary": [VERTEX_LO, VERTEX_HI],
        "_features":        features,
    }
    payload.update(all_results)

    # Write both files: canonical h3_results.json (active) and method-specific
    canonical = rdir / "h3_results.json"
    quad_out  = rdir / "h3_quadratic_results.json"
    for out in (canonical, quad_out):
        with open(out, "w") as f:
            json.dump(payload, f, indent=2,
                       default=lambda z: None if (isinstance(z, float) and np.isnan(z))
                                               else z)
        print(f"Written → {out}")


if __name__ == "__main__":
    main()
