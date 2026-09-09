#!/usr/bin/env python3
# ============================================================
# Script:  51_all_angles.py
# Release: 1.0
# Version: v1.00
# Purpose: Comprehensive comparison script. Produces every methodological
#          angle (feature variants × pooling schemes × entities) so we can
#          lock in the final methodology before rewriting the paper.
#
# Not converted to ipynb; standalone .py. Each group A..G is togglable via
# the GROUPS_ENABLED dict at the top of main().
#
# Feature variants:
#     F0:          {f0_raw, f0_norm}
#     Intensity:   {intensity_raw, intensity_norm}
#     Speech rate: {speechrate_wps, speechrate_cps}
#
# Entities:
#     HR, CZ, PL, RS, SI, GLOBAL_no_SI (4 langs), GLOBAL_with_SI (5 langs)
#     Extra pooling variants of the two globals for Group C only.
#
# Outputs:
#     results/figures/all_angles/{A..G}_*.png
#     results/all_angles.json
#     results/tables/all_angles_summary.tsv
#
# All paths resolved relative to the project root via utils.config_loader.
# ============================================================

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import (
    load_config, get_intermediate_dir, get_results_dir,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
ALL_LANGS = ["HR", "CZ", "PL", "RS", "SI"]
LANGS_4   = ["HR", "CZ", "PL", "RS"]

# Feature families, each with (raw_variant, alt_variant)
FEAT_FAMILIES = {
    "F0":          ["f0_raw", "f0_norm"],
    "Intensity":   ["intensity_raw", "intensity_norm"],
    "SpeechRate":  ["speechrate_wps", "speechrate_cps"],
}
FLAT_FEATS = [f for pair in FEAT_FAMILIES.values() for f in pair]

FEAT_LABELS = {
    "f0_raw":         "F0 (Hz)",
    "f0_norm":        "F0 (per-speaker z)",
    "intensity_raw":  "Intensity (dB)",
    "intensity_norm": "Intensity (per-session z)",
    "speechrate_wps": "Speech rate (words/s)",
    "speechrate_cps": "Speech rate (chars/s)",
}

PALETTE = {
    "HR": "#E63946", "CZ": "#457B9D", "PL": "#2A9D8F",
    "RS": "#F49F1C", "SI": "#8E44AD",
    "GLOBAL_no_SI":   "#111111",
    "GLOBAL_with_SI": "#666666",
    "eq":  "#E63946",   "sw": "#457B9D",   "mm": "#2A9D8F",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def bin_centers(n_bins: int, lo: float = 0.0, hi: float = 5.0) -> np.ndarray:
    edges = np.linspace(lo, hi, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def add_bin_col(df: pd.DataFrame, n_bins: int, score_col: str = "sentiment_score"
                ) -> pd.DataFrame:
    df = df.copy()
    df["bin"] = pd.cut(df[score_col],
                        bins=np.linspace(0, 5, n_bins + 1),
                        labels=False, include_lowest=True)
    return df


def speaker_averaged_curve(df: pd.DataFrame, feat: str, n_bins: int) -> np.ndarray:
    """Mean across speakers of each speaker's per-bin mean. Length n_bins."""
    if feat not in df.columns:
        return np.full(n_bins, np.nan)
    curves = []
    for _spk, grp in df.dropna(subset=[feat]).groupby("speaker_id"):
        c = grp.groupby("bin")[feat].mean().reindex(range(n_bins)).values
        curves.append(c)
    if not curves:
        return np.full(n_bins, np.nan)
    return np.nanmean(curves, axis=0)


def speaker_weighted_pool(dfs: list, feat: str, n_bins: int) -> np.ndarray:
    """Concatenate speaker curves across a list of language dataframes,
    then average — larger languages dominate."""
    all_curves = []
    for df in dfs:
        if feat not in df.columns:
            continue
        for _spk, grp in df.dropna(subset=[feat]).groupby("speaker_id"):
            c = grp.groupby("bin")[feat].mean().reindex(range(n_bins)).values
            all_curves.append(c)
    if not all_curves:
        return np.full(n_bins, np.nan)
    return np.nanmean(all_curves, axis=0)


def equal_language_pool(lang_curves: dict, n_bins: int) -> np.ndarray:
    """Mean of per-language speaker-averaged curves. Each language = 1 vote."""
    stack = [c for c in lang_curves.values() if np.isfinite(c).any()]
    if not stack:
        return np.full(n_bins, np.nan)
    return np.nanmean(stack, axis=0)


def minmax_normalise(curve: np.ndarray) -> np.ndarray:
    v = curve.copy()
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return v * np.nan
    return (v - lo) / (hi - lo)


def paper_normalised_global(lang_dfs: dict, feats: list, n_bins: int) -> np.ndarray:
    """Paper Figure-3 method: per-speaker curves → min-max per feature → mean
    per language → mean across languages (equal_language)."""
    lang_trends: dict[str, np.ndarray] = {}
    for lang, df in lang_dfs.items():
        per_feat_lang = []
        for feat in feats:
            if feat not in df.columns:
                continue
            spk_normed = []
            for _spk, grp in df.dropna(subset=[feat]).groupby("speaker_id"):
                c = grp.groupby("bin")[feat].mean().reindex(range(n_bins)).values
                if np.isfinite(c).sum() < 5:
                    continue
                lo, hi = np.nanmin(c), np.nanmax(c)
                if hi == lo:
                    continue
                spk_normed.append((c - lo) / (hi - lo))
            if spk_normed:
                per_feat_lang.append(np.nanmean(spk_normed, axis=0))
        if per_feat_lang:
            lang_trends[lang] = np.nanmean(per_feat_lang, axis=0)
    if not lang_trends:
        return np.full(n_bins, np.nan)
    return np.nanmean(list(lang_trends.values()), axis=0)


def ols_quadratic(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = β₀ + β₁ x + β₂ x². Return coefficients, p-values, R², vertex."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(y)
    if n < 5:
        return {k: None for k in
                ("b0", "b1", "b2", "p_b1", "p_b2", "r2_lin",
                 "r2_quad", "vertex_s", "shape", "n")}
    X = np.column_stack([np.ones(n), x, x * x])
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return {k: None for k in
                ("b0", "b1", "b2", "p_b1", "p_b2", "r2_lin",
                 "r2_quad", "vertex_s", "shape", "n")}
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
    vertex_s = (-b1 / (2 * b2)) if abs(b2) > 1e-12 else np.inf
    if p_b2 < 0.05 and b2 > 0:   shape = "U"
    elif p_b2 < 0.05 and b2 < 0: shape = "∩"
    else:                        shape = "linear"
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


def fmt_p(p) -> str:
    if p is None or not np.isfinite(p): return "n/a"
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def annotate_quad(ax, x: np.ndarray, y: np.ndarray, q: dict, color="#457B9D"):
    if q.get("b0") is None:
        return
    yhat = q["b0"] + q["b1"] * x + q["b2"] * x ** 2
    ax.plot(x, yhat, color=color, lw=1.1, ls="--",
            label=f"quad ({q['shape']})")
    if (q["p_b2"] < 0.05 and np.isfinite(q["vertex_s"])
            and 0 <= q["vertex_s"] <= 5):
        ax.axvline(q["vertex_s"], color="#888888", lw=0.8, ls=":",
                    label=f"vertex {q['vertex_s']:+.2f}")


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────
def load_features(idir: Path, langs: list, n_bins: int) -> dict:
    """Return {lang: dataframe with bin column}."""
    out = {}
    for lang in langs:
        fp = idir / f"{lang}_features.tsv"
        if not fp.exists():
            print(f"[{lang}] features TSV missing — skipping"); continue
        df = pd.read_csv(fp, sep="\t")
        df = add_bin_col(df, n_bins)
        out[lang] = df
        print(f"[{lang}] {len(df):,} utterances loaded")
    return out


def load_vad(idir: Path, langs: list) -> dict:
    """Return {lang: DataFrame with columns utterance_id, valence, arousal, dominance}."""
    out = {}
    for lang in langs:
        fp = idir / f"{lang}_vad.tsv"
        if not fp.exists():
            print(f"[{lang}] VAD TSV missing — skipping"); continue
        df = pd.read_csv(fp, sep="\t")
        out[lang] = df
    return out


# ─────────────────────────────────────────────
# Group A: 6 variants per entity (2×3 grids)
# ─────────────────────────────────────────────
def group_A(lang_dfs: dict, n_bins: int, outdir: Path, results: dict) -> None:
    """One 2×3 figure per entity showing all 6 feature variants + quadratic fit."""
    print("\n=== Group A · all 6 variants per entity (2×3 grids) ===")
    x = bin_centers(n_bins)
    entities: list = []
    for L in ALL_LANGS:
        if L in lang_dfs:
            entities.append((L, {L: lang_dfs[L]}))
    if all(l in lang_dfs for l in LANGS_4):
        entities.append(("GLOBAL_no_SI",   {l: lang_dfs[l] for l in LANGS_4}))
    if all(l in lang_dfs for l in ALL_LANGS):
        entities.append(("GLOBAL_with_SI", {l: lang_dfs[l] for l in ALL_LANGS}))

    order_grid = [["f0_raw", "intensity_raw", "speechrate_wps"],
                  ["f0_norm", "intensity_norm", "speechrate_cps"]]

    for entity_name, entity_langs in entities:
        # Build the per-feature curve for this entity: single language -> speaker_averaged_curve;
        # global -> equal_language pool of per-language speaker_averaged curves.
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), squeeze=False)
        fig.suptitle(f"Group A · {entity_name} · six feature variants (2×3)  "
                     f"·  curves + quadratic fits", fontsize=13)
        results.setdefault("A", {})[entity_name] = {}
        for row_i, row in enumerate(order_grid):
            for col_i, feat in enumerate(row):
                ax = axes[row_i][col_i]
                if len(entity_langs) == 1:
                    df = next(iter(entity_langs.values()))
                    y = speaker_averaged_curve(df, feat, n_bins)
                else:
                    lang_curves = {l: speaker_averaged_curve(d, feat, n_bins)
                                     for l, d in entity_langs.items()}
                    y = equal_language_pool(lang_curves, n_bins)
                q = ols_quadratic(x, y)
                results["A"][entity_name][feat] = q
                if not np.isfinite(y).any():
                    ax.set_title(f"{FEAT_LABELS[feat]}\n(no data)"); continue
                ax.plot(x, y, color=PALETTE.get(entity_name, "#F49F1C"), lw=1.8,
                         label="curve")
                annotate_quad(ax, x, y, q)
                ax.axvline(3.5, color="#BBBBBB", lw=0.8, ls=":",
                            label="paper 3.5")
                ax.set_title(f"{FEAT_LABELS[feat]}\n"
                             f"β₂={q['b2']:+.3g}  p={fmt_p(q['p_b2'])}  "
                             f"R²lin={q['r2_lin']:+.3f} R²quad={q['r2_quad']:+.3f}  "
                             f"({q['shape']})", fontsize=9)
                ax.set_xlabel("Sentiment"); ax.set_ylabel(feat, fontsize=8)
                ax.legend(fontsize=7, frameon=False, loc="best")
        plt.tight_layout()
        fig.savefig(outdir / f"A_{entity_name}_6variants.png", dpi=140)
        plt.close(fig)
    print(f"  Group A → {len(entities)} figures")


# ─────────────────────────────────────────────
# Group B: cross-entity comparison per feature variant
# ─────────────────────────────────────────────
def group_B(lang_dfs: dict, n_bins: int, outdir: Path, results: dict) -> None:
    """One figure per feature variant: all 5 langs + 2 globals overlaid (min-max normalised)."""
    print("\n=== Group B · cross-entity per feature variant ===")
    x = bin_centers(n_bins)
    results.setdefault("B", {})
    for feat in FLAT_FEATS:
        fig, ax = plt.subplots(figsize=(11, 5))
        for L in ALL_LANGS:
            if L not in lang_dfs: continue
            y = speaker_averaged_curve(lang_dfs[L], feat, n_bins)
            yn = minmax_normalise(y)
            ax.plot(x, yn, color=PALETTE[L], lw=1.3, alpha=0.85, label=L)
        for name, subset in [("GLOBAL_no_SI", LANGS_4),
                             ("GLOBAL_with_SI", ALL_LANGS)]:
            if not all(l in lang_dfs for l in subset): continue
            lang_curves = {l: speaker_averaged_curve(lang_dfs[l], feat, n_bins)
                             for l in subset}
            y = equal_language_pool(lang_curves, n_bins)
            yn = minmax_normalise(y)
            ax.plot(x, yn, color=PALETTE[name], lw=2.4, alpha=0.95,
                     ls="--", label=name)
        ax.axvline(3.5, color="#DDDDDD", lw=0.8, ls=":", label="paper 3.5")
        ax.set_xlabel("Sentiment  (0 = Negative → 5 = Positive)")
        ax.set_ylabel("Feature value (min-max normalised per curve)")
        ax.set_title(f"Group B · Cross-entity trends · {FEAT_LABELS[feat]}   "
                     f"(min-max normalised for shape comparison)")
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
        plt.tight_layout()
        fig.savefig(outdir / f"B_{feat}_cross_entity.png", dpi=140)
        plt.close(fig)
    print(f"  Group B → {len(FLAT_FEATS)} figures")


# ─────────────────────────────────────────────
# Group C: pooling method comparison
# ─────────────────────────────────────────────
def group_C(lang_dfs: dict, n_bins: int, outdir: Path, results: dict) -> None:
    """One figure per feature family × variant showing 6 curves:
        {equal-lang, speaker-weighted, min-max paper} × {no_SI, with_SI}.
    All plotted on min-max normalised y-axis for shape comparability."""
    print("\n=== Group C · pooling method comparison ===")
    x = bin_centers(n_bins)
    results.setdefault("C", {})
    for family, feats in FEAT_FAMILIES.items():
        for feat in feats:
            fig, ax = plt.subplots(figsize=(11, 5))
            results["C"].setdefault(family, {}).setdefault(feat, {})
            for subset_name, subset in [("no_SI", LANGS_4),
                                          ("with_SI", ALL_LANGS)]:
                if not all(l in lang_dfs for l in subset): continue
                lang_curves = {l: speaker_averaged_curve(lang_dfs[l], feat, n_bins)
                                 for l in subset}
                # Equal-language
                y_eq = equal_language_pool(lang_curves, n_bins)
                # Speaker-weighted
                y_sw = speaker_weighted_pool([lang_dfs[l] for l in subset], feat, n_bins)
                # Min-max paper (single feature; per-speaker min-max → per-lang → across-lang)
                y_mm = paper_normalised_global({l: lang_dfs[l] for l in subset},
                                                 [feat], n_bins)
                ls = "-" if subset_name == "with_SI" else "--"
                ax.plot(x, minmax_normalise(y_eq), color=PALETTE["eq"], ls=ls, lw=1.7,
                         label=f"equal-lang · {subset_name}")
                ax.plot(x, minmax_normalise(y_sw), color=PALETTE["sw"], ls=ls, lw=1.7,
                         label=f"speaker-weighted · {subset_name}")
                ax.plot(x, y_mm,                     color=PALETTE["mm"], ls=ls, lw=1.7,
                         label=f"paper min-max · {subset_name}")
                # Quadratic fits recorded (not overlaid, to keep the plot readable)
                results["C"][family][feat][subset_name] = {
                    "equal_lang":         ols_quadratic(x, y_eq),
                    "speaker_weighted":   ols_quadratic(x, y_sw),
                    "minmax_paper":       ols_quadratic(x, y_mm),
                }
            ax.axvline(3.5, color="#DDDDDD", lw=0.8, ls=":", label="paper 3.5")
            ax.set_xlabel("Sentiment"); ax.set_ylabel("Curve (min-max normalised)")
            ax.set_title(f"Group C · Pooling comparison · {FEAT_LABELS[feat]}  "
                         f"(equal-lang / speaker-weighted / paper min-max, "
                         f"with-SI solid vs no-SI dashed)")
            ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
            plt.tight_layout()
            fig.savefig(outdir / f"C_{family}_{feat}_pooling.png", dpi=140)
            plt.close(fig)
    print(f"  Group C → {len(FLAT_FEATS)} figures")


# ─────────────────────────────────────────────
# Group D: quadratic-AH summary heatmap
# ─────────────────────────────────────────────
def group_D(lang_dfs: dict, n_bins: int, outdir: Path, results: dict) -> None:
    print("\n=== Group D · quadratic-AH summary heatmap ===")
    x = bin_centers(n_bins)
    rows = []
    for L in ALL_LANGS:
        if L in lang_dfs: rows.append((L, lang_dfs[L], None))
    if all(l in lang_dfs for l in LANGS_4):
        rows.append(("GLOBAL_no_SI",   None, LANGS_4))
    if all(l in lang_dfs for l in ALL_LANGS):
        rows.append(("GLOBAL_with_SI", None, ALL_LANGS))

    matrix_b2 = np.full((len(rows), len(FLAT_FEATS)), np.nan)
    matrix_p  = np.full((len(rows), len(FLAT_FEATS)), np.nan)
    shape_str = np.full((len(rows), len(FLAT_FEATS)), "", dtype=object)
    verts     = np.full((len(rows), len(FLAT_FEATS)), np.nan)

    results.setdefault("D", {"rows": [r[0] for r in rows], "cols": FLAT_FEATS,
                             "cells": {}})
    for i, (name, df, subset) in enumerate(rows):
        for j, feat in enumerate(FLAT_FEATS):
            if df is not None:
                y = speaker_averaged_curve(df, feat, n_bins)
            else:
                lang_curves = {l: speaker_averaged_curve(lang_dfs[l], feat, n_bins)
                                 for l in subset}
                y = equal_language_pool(lang_curves, n_bins)
            q = ols_quadratic(x, y)
            results["D"]["cells"][f"{name}::{feat}"] = q
            if q.get("b2") is None: continue
            matrix_b2[i, j] = q["b2"]
            matrix_p[i, j]  = q["p_b2"]
            shape_str[i, j] = q["shape"]
            verts[i, j]     = q["vertex_s"]

    # Symmetric colour scale
    vmax = float(np.nanmax(np.abs(matrix_b2))) if np.isfinite(matrix_b2).any() else 1.0
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(FLAT_FEATS) + 4),
                                     0.5 * len(rows) + 3))
    im = ax.imshow(matrix_b2, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(FLAT_FEATS)):
            if not np.isfinite(matrix_b2[i, j]): continue
            txt = f"{matrix_b2[i, j]:+.3g}\n{shape_str[i, j]}  p={fmt_p(matrix_p[i, j])}"
            if np.isfinite(verts[i, j]) and 0 <= verts[i, j] <= 5 and matrix_p[i, j] < 0.05:
                txt += f"\nvertex {verts[i, j]:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                     color="black")
    ax.set_xticks(range(len(FLAT_FEATS)))
    ax.set_xticklabels([FEAT_LABELS[f] for f in FLAT_FEATS], rotation=25, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_title("Group D · Quadratic-AH β₂ summary  "
                 "(red = U-shape / positive β₂; blue = ∩ / negative β₂)")
    fig.colorbar(im, ax=ax, label="β₂  (positive → U-shape)")
    plt.tight_layout()
    fig.savefig(outdir / "D_quadratic_ah_heatmap.png", dpi=140)
    plt.close(fig)
    print("  Group D → 1 figure")


# ─────────────────────────────────────────────
# Group E: sentiment-acoustic Kendall correlation heatmap
# ─────────────────────────────────────────────
def group_E(lang_dfs: dict, n_bins: int, outdir: Path, results: dict) -> None:
    print("\n=== Group E · sentiment-acoustic Kendall correlation heatmap ===")
    x = bin_centers(n_bins)
    rows = []
    for L in ALL_LANGS:
        if L in lang_dfs: rows.append((L, lang_dfs[L], None))
    if all(l in lang_dfs for l in LANGS_4):
        rows.append(("GLOBAL_no_SI",   None, LANGS_4))
    if all(l in lang_dfs for l in ALL_LANGS):
        rows.append(("GLOBAL_with_SI", None, ALL_LANGS))

    kmat = np.full((len(rows), len(FLAT_FEATS)), np.nan)
    pmat = np.full((len(rows), len(FLAT_FEATS)), np.nan)
    results.setdefault("E", {"rows": [r[0] for r in rows], "cols": FLAT_FEATS,
                             "cells": {}})
    for i, (name, df, subset) in enumerate(rows):
        for j, feat in enumerate(FLAT_FEATS):
            if df is not None:
                y = speaker_averaged_curve(df, feat, n_bins)
            else:
                lang_curves = {l: speaker_averaged_curve(lang_dfs[l], feat, n_bins)
                                 for l in subset}
                y = equal_language_pool(lang_curves, n_bins)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 5: continue
            tau, p = stats.kendalltau(x[m], y[m])
            kmat[i, j] = tau
            pmat[i, j] = p
            results["E"]["cells"][f"{name}::{feat}"] = {
                "kendall_tau": float(tau), "kendall_p": float(p), "n": int(m.sum())}

    vmax = float(np.nanmax(np.abs(kmat))) if np.isfinite(kmat).any() else 1.0
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(FLAT_FEATS) + 4),
                                     0.5 * len(rows) + 3))
    im = ax.imshow(kmat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(FLAT_FEATS)):
            if not np.isfinite(kmat[i, j]): continue
            ax.text(j, i, f"{kmat[i, j]:+.3f}\np={fmt_p(pmat[i, j])}",
                     ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(FLAT_FEATS)))
    ax.set_xticklabels([FEAT_LABELS[f] for f in FLAT_FEATS], rotation=25, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_title("Group E · Kendall τ  (bin-level, speaker-averaged curve)  "
                 "sentiment × acoustic feature  (red = negative τ)")
    fig.colorbar(im, ax=ax, label="Kendall τ")
    plt.tight_layout()
    fig.savefig(outdir / "E_sentiment_acoustic_kendall.png", dpi=140)
    plt.close(fig)
    print("  Group E → 1 figure")


# ─────────────────────────────────────────────
# Group F: NRC-VAD text correlations
# ─────────────────────────────────────────────
def group_F(lang_dfs: dict, vad_dfs: dict, outdir: Path, results: dict) -> None:
    """For each language×acoustic feature × text-VAD dim: Spearman r utterance-level.
    One heatmap per VAD dim; also a per-lang×dim correlation with sentiment."""
    print("\n=== Group F · NRC-VAD text × acoustics correlations ===")
    if not vad_dfs:
        print("  no VAD TSVs — skipping"); return
    results.setdefault("F", {})
    text_dims = ["valence", "arousal", "dominance"]

    # (F1) heatmap per dim: rows=lang, cols=acoustic feature variants
    for dim in text_dims:
        rmat = np.full((len(ALL_LANGS), len(FLAT_FEATS)), np.nan)
        pmat = np.full((len(ALL_LANGS), len(FLAT_FEATS)), np.nan)
        for i, L in enumerate(ALL_LANGS):
            if L not in lang_dfs or L not in vad_dfs: continue
            merged = lang_dfs[L].merge(
                vad_dfs[L][["utterance_id", dim]], on="utterance_id", how="inner"
            )
            for j, feat in enumerate(FLAT_FEATS):
                if feat not in merged.columns: continue
                sub = merged.dropna(subset=[feat, dim])
                if len(sub) < 30: continue
                r, p = stats.spearmanr(sub[feat], sub[dim])
                rmat[i, j] = r; pmat[i, j] = p
                results["F"].setdefault(L, {}).setdefault(feat, {})[dim] = {
                    "spearman_r": float(r), "spearman_p": float(p), "n": int(len(sub))}
        vmax = float(np.nanmax(np.abs(rmat))) if np.isfinite(rmat).any() else 1.0
        fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(FLAT_FEATS) + 4),
                                         0.5 * len(ALL_LANGS) + 3))
        im = ax.imshow(rmat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(ALL_LANGS)):
            for j in range(len(FLAT_FEATS)):
                if not np.isfinite(rmat[i, j]): continue
                ax.text(j, i, f"{rmat[i, j]:+.3f}\np={fmt_p(pmat[i, j])}",
                         ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(FLAT_FEATS)))
        ax.set_xticklabels([FEAT_LABELS[f] for f in FLAT_FEATS], rotation=25, ha="right")
        ax.set_yticks(range(len(ALL_LANGS))); ax.set_yticklabels(ALL_LANGS)
        ax.set_title(f"Group F · NRC text {dim} × acoustic (utterance-level Spearman r)")
        fig.colorbar(im, ax=ax, label="Spearman r")
        plt.tight_layout()
        fig.savefig(outdir / f"F_nrc_{dim}_vs_acoustic.png", dpi=140)
        plt.close(fig)

    # (F2) sentiment × VAD dim correlations (single bar+heatmap panel)
    sent_r = np.full((len(ALL_LANGS), len(text_dims)), np.nan)
    sent_p = np.full((len(ALL_LANGS), len(text_dims)), np.nan)
    for i, L in enumerate(ALL_LANGS):
        if L not in lang_dfs or L not in vad_dfs: continue
        merged = lang_dfs[L].merge(
            vad_dfs[L][["utterance_id"] + text_dims], on="utterance_id", how="inner"
        )
        for j, dim in enumerate(text_dims):
            sub = merged.dropna(subset=["sentiment_score", dim])
            if len(sub) < 30: continue
            r, p = stats.spearmanr(sub["sentiment_score"], sub[dim])
            sent_r[i, j] = r; sent_p[i, j] = p
            results["F"].setdefault(f"{L}_sentiment_vs_vad", {})[dim] = {
                "spearman_r": float(r), "spearman_p": float(p), "n": int(len(sub))}
    vmax = float(np.nanmax(np.abs(sent_r))) if np.isfinite(sent_r).any() else 1.0
    fig, ax = plt.subplots(figsize=(5, 0.5 * len(ALL_LANGS) + 3))
    im = ax.imshow(sent_r, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(ALL_LANGS)):
        for j in range(len(text_dims)):
            if not np.isfinite(sent_r[i, j]): continue
            ax.text(j, i, f"{sent_r[i, j]:+.3f}\np={fmt_p(sent_p[i, j])}",
                     ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(text_dims))); ax.set_xticklabels(text_dims)
    ax.set_yticks(range(len(ALL_LANGS))); ax.set_yticklabels(ALL_LANGS)
    ax.set_title("Group F · ParlaSent sentiment × NRC-VAD (utterance Spearman)")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(outdir / "F_sentiment_vs_nrc_vad.png", dpi=140)
    plt.close(fig)

    # ─────────────────────────────────────────
    # F3 · Three showcase Spearman correlations
    #      per language + one pooled GLOBAL bar
    # ─────────────────────────────────────────
    def _corr_pair(merged: pd.DataFrame, feat: str, dim: str) -> tuple:
        sub = merged.dropna(subset=[feat, dim])
        if len(sub) < 30: return (np.nan, np.nan, 0)
        r, p = stats.spearmanr(sub[feat], sub[dim])
        return (float(r), float(p), int(len(sub)))

    # Build per-lang merges once so all 3 showcase pairs share the same merged frames
    merges = {}
    for L in ALL_LANGS:
        if L in lang_dfs and L in vad_dfs:
            merges[L] = lang_dfs[L].merge(
                vad_dfs[L][["utterance_id"] + text_dims],
                on="utterance_id", how="inner"
            )
    # Pooled GLOBAL frame: concatenate utterance rows across langs
    if merges:
        pooled = pd.concat(list(merges.values()), ignore_index=True)
    else:
        pooled = None

    _showcase = [
        ("sentiment × NRC-valence",
         [("sentiment_score", "valence")],
         "sentiment_score"),
        ("intensity × NRC-arousal",
         [("intensity_raw", "arousal"), ("intensity_norm", "arousal")],
         None),
        ("NRC-dominance × speech rate",
         [("speechrate_wps", "dominance"), ("speechrate_cps", "dominance")],
         None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
    axes = axes[0]
    results["F"].setdefault("showcase", {})
    for ax, (title, pairs, _) in zip(axes, _showcase):
        # Each subplot: grouped bars over LANGS + GLOBAL; one group per (feat, dim) pair
        n_groups = len(ALL_LANGS) + 1
        n_pairs  = len(pairs)
        width = 0.8 / n_pairs
        xbase = np.arange(n_groups)
        for pi, (feat, dim) in enumerate(pairs):
            heights, texts = [], []
            for L in ALL_LANGS:
                if L not in merges:
                    heights.append(np.nan); texts.append(""); continue
                r, p, n = _corr_pair(merges[L], feat, dim)
                heights.append(r)
                texts.append(f"r={r:+.3f}\np={fmt_p(p)}\nn={n:,}")
                results["F"]["showcase"].setdefault(f"{feat}_vs_{dim}", {})[L] = {
                    "spearman_r": r, "spearman_p": p, "n": n}
            # GLOBAL pooled
            if pooled is not None:
                r, p, n = _corr_pair(pooled, feat, dim)
                heights.append(r)
                texts.append(f"r={r:+.3f}\np={fmt_p(p)}\nn={n:,}")
                results["F"]["showcase"].setdefault(f"{feat}_vs_{dim}", {})["GLOBAL_pooled"] = {
                    "spearman_r": r, "spearman_p": p, "n": n}
            else:
                heights.append(np.nan); texts.append("")
            xs = xbase - 0.4 + width * (pi + 0.5)
            colors = ["#E63946" if pi == 0 else "#457B9D"] * n_groups
            bars = ax.bar(xs, heights, width * 0.9,
                          color=colors, edgecolor="black",
                          label=f"{FEAT_LABELS.get(feat, feat)} × {dim}")
            for xi, hi, tx in zip(xs, heights, texts):
                if not np.isfinite(hi): continue
                y_off = 0.008 if hi >= 0 else -0.008
                ax.text(xi, hi + y_off, tx, ha="center",
                        va="bottom" if hi >= 0 else "top", fontsize=6)
        ax.axhline(0, color="#000", lw=0.8)
        ax.set_xticks(xbase)
        ax.set_xticklabels(ALL_LANGS + ["GLOBAL"], rotation=0)
        ax.set_ylabel("Spearman r  (utterance-level)")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("Group F3 · Showcase Spearman correlations "
                 "(per-language + pooled GLOBAL utterance-level)", fontsize=12)
    plt.tight_layout()
    fig.savefig(outdir / "F_showcase_correlations.png", dpi=140)
    plt.close(fig)

    # ─────────────────────────────────────────
    # F4 · Auto-scan every (target × VAD) pair
    #      per language + pooled GLOBAL. Rank by |r|.
    # ─────────────────────────────────────────
    scan_targets = ["sentiment_score"] + FLAT_FEATS
    results["F"].setdefault("scan", {})
    def _scan(merged: pd.DataFrame) -> list:
        rows = []
        for tgt in scan_targets:
            if tgt not in merged.columns: continue
            for dim in text_dims:
                r, p, n = _corr_pair(merged, tgt, dim)
                if not np.isfinite(r): continue
                rows.append({"target": tgt, "dim": dim,
                             "r": r, "p": p, "n": n, "abs_r": abs(r)})
        rows.sort(key=lambda d: d["abs_r"], reverse=True)
        return rows

    # Per-language scan → heatmap + ranked JSON
    scan_by_lang = {}
    for L in ALL_LANGS:
        if L not in merges: continue
        rows = _scan(merges[L])
        scan_by_lang[L] = rows
        results["F"]["scan"][L] = rows
        top = rows[:5]
        print(f"\n  [F4] {L} — top 5 |r| across (target × VAD dim):")
        for row in top:
            print(f"    {row['target']:<20s} × {row['dim']:9s}  "
                  f"r={row['r']:+.3f}  p={fmt_p(row['p'])}  n={row['n']:,}")

    # Pooled GLOBAL scan
    if pooled is not None:
        rows = _scan(pooled)
        scan_by_lang["GLOBAL_pooled"] = rows
        results["F"]["scan"]["GLOBAL_pooled"] = rows
        print(f"\n  [F4] GLOBAL_pooled — top 5 |r|:")
        for row in rows[:5]:
            print(f"    {row['target']:<20s} × {row['dim']:9s}  "
                  f"r={row['r']:+.3f}  p={fmt_p(row['p'])}  n={row['n']:,}")

    # Heatmap: rows = LANGS + GLOBAL, cols = target × dim (flattened, ordered by scan_targets first)
    pair_cols = [(t, d) for t in scan_targets for d in text_dims]
    row_labels = [L for L in ALL_LANGS if L in scan_by_lang] + \
                 (["GLOBAL_pooled"] if "GLOBAL_pooled" in scan_by_lang else [])
    R = np.full((len(row_labels), len(pair_cols)), np.nan)
    P = np.full_like(R, np.nan)
    for i, L in enumerate(row_labels):
        by_pair = {(row["target"], row["dim"]): row for row in scan_by_lang[L]}
        for j, (tgt, dim) in enumerate(pair_cols):
            row = by_pair.get((tgt, dim))
            if row:
                R[i, j] = row["r"]; P[i, j] = row["p"]
    vmax = float(np.nanmax(np.abs(R))) if np.isfinite(R).any() else 1.0
    fig, ax = plt.subplots(figsize=(max(12, 0.7 * len(pair_cols) + 2),
                                     0.5 * len(row_labels) + 3))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    # Annotate cells; bold-mark each row's max |r|
    for i in range(len(row_labels)):
        row_absmax_j = int(np.nanargmax(np.abs(R[i]))) if np.isfinite(R[i]).any() else -1
        for j in range(len(pair_cols)):
            if not np.isfinite(R[i, j]): continue
            weight = "bold" if j == row_absmax_j else "normal"
            ax.text(j, i,
                     f"{R[i, j]:+.3f}\np={fmt_p(P[i, j])}",
                     ha="center", va="center", fontsize=6,
                     fontweight=weight)
    ax.set_xticks(range(len(pair_cols)))
    ax.set_xticklabels([f"{t}\n×{d[0].upper()}" for t, d in pair_cols],
                        rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_title("Group F4 · Auto-scan: Spearman r for every "
                 "(target × VAD dim) pair  ·  bold = row's max |r|")
    fig.colorbar(im, ax=ax, label="Spearman r")
    plt.tight_layout()
    fig.savefig(outdir / "F_scan_all_pairs.png", dpi=140)
    plt.close(fig)
    print("  Group F → 6 figures  (V/A/D × acoustic + sentiment × VAD + showcase + scan)")


# ─────────────────────────────────────────────
# Group G: VH1 concordance bar plot (from h1_results.json)
# ─────────────────────────────────────────────
def group_G(rdir: Path, outdir: Path, results: dict) -> None:
    h1_path = rdir / "h1_results.json"
    if not h1_path.exists():
        print("  h1_results.json missing — skipping Group G"); return
    print("\n=== Group G · VH1 concordance bar plot ===")
    with open(h1_path) as f: h1 = json.load(f)
    main_feats = ["f0_raw", "intensity_norm", "speechrate_wps"]  # match features_main
    labels = ALL_LANGS
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    results.setdefault("G", {})
    for k, feat in enumerate(main_feats):
        vals, pvals = [], []
        for L in labels:
            key = f"{L}_{feat}"
            v = h1.get(key, {}).get("speaker_avg", {})
            vals.append(v.get("concordance", np.nan))
            pvals.append(v.get("p", np.nan))
            results["G"].setdefault(L, {})[feat] = {
                "concordance": v.get("concordance"),
                "p": v.get("p"), "rbc": v.get("rbc"), "n": v.get("n")}
        offsets = (k - 1) * width
        for xi, xv, pv in zip(x + offsets, vals, pvals):
            hatch = "///" if (pv is not None and pv >= 0.05) else None
            ax.bar(xi, xv, width=width * 0.95,
                    color=PALETTE.get({"f0_raw":"HR","intensity_norm":"CZ",
                                       "speechrate_wps":"PL"}[feat], "#888"),
                    edgecolor="black", hatch=hatch,
                    label=FEAT_LABELS[feat] if xi == x[0] + offsets else None)
    ax.axhline(0.5, color="#888", lw=0.8, ls="--", label="chance (0.5)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("P(Neg > Pos)  (speaker-averaged concordance)")
    ax.set_title("Group G · VH1 concordance probability across languages  "
                 "(hashed = n.s. at p ≥ 0.05)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    plt.tight_layout()
    fig.savefig(outdir / "G_vh1_concordance.png", dpi=140)
    plt.close(fig)
    print("  Group G → 1 figure")


# ─────────────────────────────────────────────
# Summary TSV
# ─────────────────────────────────────────────
def write_summary_tsv(results: dict, out_tsv: Path) -> None:
    rows = []
    A = results.get("A", {})
    E = results.get("E", {}).get("cells", {})
    for entity, feats in A.items():
        for feat, q in feats.items():
            e_key = f"{entity}::{feat}"
            e_cell = E.get(e_key, {})
            rows.append({
                "entity":     entity,
                "feature":    feat,
                "n":          q.get("n"),
                "b1":         q.get("b1"),
                "b2":         q.get("b2"),
                "p_b1":       q.get("p_b1"),
                "p_b2":       q.get("p_b2"),
                "vertex_s":   q.get("vertex_s"),
                "shape":      q.get("shape"),
                "r2_lin":     q.get("r2_lin"),
                "r2_quad":    q.get("r2_quad"),
                "kendall_tau":e_cell.get("kendall_tau"),
                "kendall_p":  e_cell.get("kendall_p"),
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
        print(f"\nSummary TSV → {out_tsv}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--outdir", default=None,
                   help="Override output subdir (default: results/figures/all_angles)")
    p.add_argument("--groups", default="ABCDEFG",
                   help="Which groups to run (subset of ABCDEFG, e.g. --groups AD)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    idir = get_intermediate_dir(cfg)
    rdir = get_results_dir(cfg)
    n_bins = cfg["analysis"]["n_bins"]

    figdir = Path(args.outdir) if args.outdir else (rdir / "figures" / "all_angles")
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir = rdir / "tables"; tabdir.mkdir(parents=True, exist_ok=True)

    GROUPS_ENABLED = {g: (g in args.groups) for g in "ABCDEFG"}
    print(f"Groups enabled: {[g for g,on in GROUPS_ENABLED.items() if on]}")
    print(f"Output figures  → {figdir}")
    print(f"Output tables   → {tabdir}")

    lang_dfs = load_features(idir, ALL_LANGS, n_bins)
    vad_dfs  = load_vad(idir, ALL_LANGS) if GROUPS_ENABLED["F"] else {}

    results: dict = {"n_bins": n_bins, "entities_loaded": list(lang_dfs.keys())}
    if GROUPS_ENABLED["A"]: group_A(lang_dfs, n_bins, figdir, results)
    if GROUPS_ENABLED["B"]: group_B(lang_dfs, n_bins, figdir, results)
    if GROUPS_ENABLED["C"]: group_C(lang_dfs, n_bins, figdir, results)
    if GROUPS_ENABLED["D"]: group_D(lang_dfs, n_bins, figdir, results)
    if GROUPS_ENABLED["E"]: group_E(lang_dfs, n_bins, figdir, results)
    if GROUPS_ENABLED["F"]: group_F(lang_dfs, vad_dfs, figdir, results)
    if GROUPS_ENABLED["G"]: group_G(rdir, figdir, results)

    write_summary_tsv(results, tabdir / "all_angles_summary.tsv")
    out_json = rdir / "all_angles.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x))
                                    else (float(x) if isinstance(x, np.floating)
                                          else int(x) if isinstance(x, np.integer)
                                          else str(x)))
    print(f"\nAll-angles JSON → {out_json}")


if __name__ == "__main__":
    main()
