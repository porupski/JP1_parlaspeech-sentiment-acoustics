# ============================================================
# Script:  43_editorial_publishable.py
# Release: 1.0
# Version: v1.00
# Purpose: Publication-grade editorial PNGs (v3-style: H1 + H2 only,
#          NO quadratic). Reuses utils.plotting where possible.
#
# Toggle LANGUAGES / FEATURES at the top to include/exclude subsets
# (e.g. drop "SI" from LANGUAGES to regenerate without Slovenian).
# ============================================================

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_loader import load_config, get_intermediate_dir, get_results_dir
from utils.data_utils import compute_bins, speaker_binned_means
from utils.plotting import plot_trend_curves, save_fig, PALETTE

# ===== EDITORIAL CONFIG =====================================================
LANGUAGES = ["HR", "CZ", "PL", "RS"]                  # remove any to exclude
FEATURES  = ["f0_raw", "intensity_raw", "speechrate_wps"]

OUT_DIR   = Path(__file__).resolve().parent.parent / "results/figures/publishable"
DPI       = 300

# Palette (matches the v3 editorial reference)
ORANGE = "#FFA500"     # speaker-average bars, global curve
BLUE   = "#1f77b4"     # instance-level bars
GRID   = "#D0D0D0"

LANG_FULL = {"HR": "Croatian", "CZ": "Czech", "PL": "Polish",
             "RS": "Serbian",  "SI": "Slovenian"}
FEAT_SHORT = {"f0_raw": "F0", "intensity_norm": "Int",
              "speechrate_wps": "SRwnp",
              "f0_norm": "F0norm", "intensity_raw": "IntRaw",
              "speechrate_sps": "SRsps"}
# ============================================================================

# Force palette to editorial orange for reused trend-curve plot
PALETTE["bin_curve"] = ORANGE

REPO = Path(__file__).resolve().parent.parent
cfg  = load_config(str(REPO / "config.json"))
idir = REPO / cfg["paths"]["intermediate_dir"]      # absolute
# results-final = curated frozen H1/H2 JSONs; override the config's default "results"
rdir = REPO / "results"
s_min = cfg["analysis"]["sentiment_min"]
s_max = cfg["analysis"]["sentiment_max"]
n_bins = cfg["analysis"]["n_bins"]

def _load(name): return json.load(open(rdir / name))
h1 = _load("h1_results.json")
h2 = _load("h2_results.json")

# ---------- Load per-lang binned data ---------------------------------------
binned = {}
for lang in LANGUAGES:
    tsv = idir / f"{lang}_features.tsv"
    if not tsv.exists():
        print(f"missing {tsv}, skip"); continue
    df = compute_bins(pd.read_csv(tsv, sep="\t"), n_bins=n_bins)
    for feat in FEATURES:
        if feat in df.columns:
            binned[(lang, feat)] = speaker_binned_means(df, [feat])[feat]

# ---------- helpers ---------------------------------------------------------
def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return "—"
    if p < 1e-6:  return "<1e-6"
    if p < 0.001: return f"{p:.1e}"
    return f"{p:.3f}"

# ============================================================================
# FIG 1 — Concordance probability bars (H1, speaker + utterance)
# ============================================================================
def fig1_concordance():
    n_groups = len(LANGUAGES) * len(FEATURES)
    fig_w = max(10.0, 1.3 * n_groups)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    bar_w = 0.38
    within_gap = 0.15    # between features within a language
    between_gap = 1.0    # between languages

    x_pos, x_labs, lang_centers = [], [], []
    pos = 0.0
    for lang in LANGUAGES:
        start = pos
        for feat in FEATURES:
            r  = h1.get(f"{lang}_{feat}", {})
            sa = r.get("speaker_avg", {}); ul = r.get("utterance_level", {})
            sa_c = sa.get("concordance", np.nan); ul_c = ul.get("concordance", np.nan)
            sa_ns = sa.get("p_bh", sa.get("p", 1.0)) >= 0.05
            ul_ns = ul.get("p_bh", ul.get("p", 1.0)) >= 0.05

            ax.bar(pos,           sa_c, width=bar_w, color=ORANGE,
                   hatch="///" if sa_ns else None,
                   edgecolor="black", linewidth=0.7,
                   alpha=0.55 if sa_ns else 1.0)
            ax.bar(pos + bar_w,   ul_c, width=bar_w, color=BLUE,
                   hatch="///" if ul_ns else None,
                   edgecolor="black", linewidth=0.7,
                   alpha=0.55 if ul_ns else 1.0)
            if not np.isnan(sa_c):
                ax.text(pos, sa_c + 0.015, f"{sa_c:.2f}",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            if not np.isnan(ul_c):
                ax.text(pos + bar_w, ul_c + 0.015, f"{ul_c:.2f}",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

            x_pos.append(pos + bar_w / 2)
            x_labs.append(FEAT_SHORT.get(feat, feat))
            pos += 1 + within_gap
        end = pos - 1 - within_gap
        lang_centers.append((start + end + bar_w) / 2)
        pos += between_gap

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labs, fontsize=11)
    for c, lang in zip(lang_centers, LANGUAGES):
        ax.text(c, -0.10, LANG_FULL[lang],
                ha="center", va="top", fontsize=13, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 1.18)
    ax.set_ylabel("P(Neg > Pos)", fontsize=13, fontweight="bold")
    ax.set_title("Concordance Probabilities", fontsize=15, fontweight="bold")

    handles = [
        mpatches.Patch(facecolor=ORANGE, edgecolor="black", label="Speaker average"),
        mpatches.Patch(facecolor=BLUE,   edgecolor="black", label="Instance level"),
        mpatches.Patch(facecolor="white", hatch="///", edgecolor="black",
                       label="Insignificant (p > 0.05)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "fig1_concordance.png", dpi=DPI)
    print(f"→ fig1_concordance.png")

# ============================================================================
# FIG 2 — Global average curve (all langs × all feats, min-max normalized)
# ============================================================================
def fig2_global_trend():
    normed = []
    for curve in binned.values():
        y = curve.values.astype(float)
        vmin, vmax = np.nanmin(y), np.nanmax(y)
        if vmax > vmin:
            normed.append((y - vmin) / (vmax - vmin))
    if not normed:
        print("no data"); return
    global_curve = np.nanmean(np.vstack(normed), axis=0)

    x = np.linspace(s_min, s_max, len(global_curve))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, global_curve, color=ORANGE, lw=2.5,
            marker="o", markersize=5, label="60-bin Global Average")

    lang_tag = "with SI" if "SI" in LANGUAGES else "no SI"
    ax.set_xlabel("Sentiment Logit (0 = Negative → 5 = Positive)",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalized Feature Value (0–1)",
                  fontsize=13, fontweight="bold")
    ax.set_title(f"Global Average: All Languages × All Features (Normalized, {lang_tag})",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.4, color=GRID)
    ax.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "fig2_global_trend.png", dpi=DPI)
    print(f"→ fig2_global_trend.png")

# ============================================================================
# FIG 3 — Per-language × per-feature breakout (reuses utils.plotting)
# ============================================================================
def fig3_breakout():
    fig = plot_trend_curves(
        binned, languages=LANGUAGES, features=FEATURES,
        sentiment_min=s_min, sentiment_max=s_max,
        figsize=(3.4 * len(LANGUAGES), 3.2 * len(FEATURES)),
        show_linear_fit=True, panel_hspace=0.35, panel_wspace=0.25,
        title="Sentiment × Acoustic Feature — per Language",
    )
    save_fig(fig, OUT_DIR / "fig3_breakout.png", dpi=DPI)
    print(f"→ fig3_breakout.png")

# ============================================================================
# TABLES → PNG (H1 concordance, H2 Kendall tau)
# ============================================================================
def _render_table(header, rows, path, title, col_widths=None):
    n_cols = len(header); n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(0.85 * n_cols + 2, 0.42 * (n_rows + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header,
                   colWidths=col_widths, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    for i in range(n_cols):
        c = tbl[(0, i)]
        c.set_facecolor(ORANGE); c.set_text_props(weight="bold", color="black")
    for r in range(1, n_rows + 1):
        if r % 2 == 0:
            for c_ in range(n_cols):
                tbl[(r, c_)].set_facecolor("#F7F7F7")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    save_fig(fig, path, dpi=DPI)

def table_h1():
    header = ["Lang", "Feat",
              "P(N>P) spk", "p_BH spk",
              "P(N>P) utt", "p_BH utt", "n spk"]
    rows = []
    for lang in LANGUAGES:
        for feat in FEATURES:
            r  = h1.get(f"{lang}_{feat}", {})
            sa = r.get("speaker_avg", {}); ul = r.get("utterance_level", {})
            rows.append([
                LANG_FULL[lang], FEAT_SHORT.get(feat, feat),
                f"{sa.get('concordance', float('nan')):.3f}",
                fmt_p(sa.get("p_bh")),
                f"{ul.get('concordance', float('nan')):.3f}",
                fmt_p(ul.get("p_bh")),
                str(sa.get("n", "")),
            ])
    _render_table(header, rows, OUT_DIR / "table_h1_concordance.png",
                  "H1 — Concordance P(Neg > Pos), BH-corrected")
    print(f"→ table_h1_concordance.png")

def table_h2():
    header = ["Lang", "Feat", "mean τ", "sig / n", "p_BH"]
    rows = []
    for lang in LANGUAGES:
        for feat in FEATURES:
            r = h2.get(f"{lang}_{feat}", {})
            n  = r.get("n_speakers", 0)
            ns = r.get("n_sig_speakers", 0)
            rows.append([
                LANG_FULL[lang], FEAT_SHORT.get(feat, feat),
                f"{r.get('mean_tau', float('nan')):+.3f}",
                f"{ns} / {n}",
                fmt_p(r.get("p_bh")),
            ])
    _render_table(header, rows, OUT_DIR / "table_h2_kendall.png",
                  "H2 — Kendall τ per speaker (mean, sig count, BH p)")
    print(f"→ table_h2_kendall.png")

# ============================================================================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Languages: {LANGUAGES}    Features: {FEATURES}")
    print(f"Output:    {OUT_DIR}")
    fig1_concordance()
    fig2_global_trend()
    fig3_breakout()
    table_h1()
    table_h2()
    print("done")
