# ============================================================
# Script:  42_plots_editorial.py  (jupytext % format)
# Release: 1.0
# Version: v1.00
# Purpose: Editorial/interactive plot generation.
#          Each cell is one figure. Adjust the variables at the top
#          of each cell, then re-run it. Convert to notebook with:
#              jupytext --to notebook 42_plots_editorial.py
#
# All available kwargs are documented in the docstring of each
# plotting function in utils/plotting.py.
# ============================================================

# %% [markdown]
# # Editorial Plots
# Run cells individually to tweak figures before submitting.
# Adjust variables at the top of each cell; all options are in `utils/plotting.py` docstrings.

# %% Setup
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))
from utils.config_loader import load_config, get_results_dir, get_intermediate_dir
from utils.data_utils import compute_bins, speaker_binned_means
from utils.plotting import (
    plot_concordance_bars, plot_trend_curves,
    plot_global_trend, plot_diagnostic_split, save_fig, PALETTE
)

cfg = load_config("../config.json")
langs = cfg["languages"]
feats = cfg["analysis"]["features_main"]
idir = get_intermediate_dir(cfg)
rdir = get_results_dir(cfg)
s_min = cfg["analysis"]["sentiment_min"]
s_max = cfg["analysis"]["sentiment_max"]

def _load(name):
    p = rdir / name
    return json.load(open(p)) if p.exists() else {}

def _reshape(d):
    return {tuple(k.split("_", 1)): v for k, v in d.items() if "_" in k}

# Load data
binned = {}
for lang in langs:
    path = idir / f"{lang}_features.tsv"
    if not path.exists(): continue
    df = pd.read_csv(path, sep="\t")
    df = compute_bins(df, n_bins=cfg["analysis"]["n_bins"])
    for feat in feats:
        if feat not in df.columns: continue
        binned[(lang, feat)] = speaker_binned_means(df, [feat])[feat]

h1 = _reshape(_load("h1_results.json"))
h3_raw = _reshape(_load("h3_results.json"))
gt_data = _load("global_trend.json")

split_point = gt_data.get("split_point", 3.5)
gt_series = pd.Series(
    [np.nan if v is None else v for v in gt_data.get("values", [])],
    index=gt_data.get("bins", range(cfg["analysis"]["n_bins"]))
)

print("Data loaded. Edit variables in each cell and re-run.")

# %% [markdown]
# ## Figure 1 — Concordance Probability Bars

# %% Fig 1 — Concordance bars
# ── Editorial variables ──────────────────────────────────────────────
LANGUAGES    = langs           # subset if needed, e.g. ["HR", "PL"]
FEATURES     = feats           # subset features
FIGSIZE      = (12, 5)         # (width, height) inches
FONT_SCALE   = 1.0             # 1.0 = default; try 1.2 for larger text
SHOW_CHANCE  = True            # draw 0.5 line
CHANCE_VALUE = 0.5
BAR_ALPHA    = 0.85
TITLE        = "Concordance P(Neg > Pos) — H1"
SAVE_PATH    = rdir / "figures/fig1_concordance.png"
SAVE_DPI     = 300
# ─────────────────────────────────────────────────────────────────────
# Available kwargs: figsize, font_scale, show_chance_line, chance_value,
#   bar_alpha, title, xlabel
# See utils/plotting.py → plot_concordance_bars() for full docstring.

fig = plot_concordance_bars(
    h1, languages=LANGUAGES, features=FEATURES,
    figsize=FIGSIZE, font_scale=FONT_SCALE,
    show_chance_line=SHOW_CHANCE, chance_value=CHANCE_VALUE,
    bar_alpha=BAR_ALPHA, title=TITLE,
)
save_fig(fig, SAVE_PATH, dpi=SAVE_DPI)
plt.show()

# %% [markdown]
# ## Figure 2 — Sentiment–Acoustic Trend Curves

# %% Fig 2 — Trend curves
# ── Editorial variables ──────────────────────────────────────────────
LANGUAGES      = langs
FEATURES       = feats
FIGSIZE        = None          # None = auto (18 × n_features × 3.2)
FONT_SCALE     = 1.0
SHOW_LINEAR    = True          # overlay full-range dashed linear fit
SHOW_CI_BAND   = False         # shaded bootstrap CI (requires ci_data dict)
PANEL_HSPACE   = 0.35          # vertical spacing between feature rows
PANEL_WSPACE   = 0.25          # horizontal spacing between language cols
TITLE          = "Sentiment–Acoustic Trends (H2)"
SAVE_PATH      = rdir / "figures/fig2_trends.png"
SAVE_DPI       = 300
# ─────────────────────────────────────────────────────────────────────
# Available kwargs: figsize, font_scale, show_linear_fit, show_ci_band,
#   ci_data, panel_hspace, panel_wspace, title
# See utils/plotting.py → plot_trend_curves() for full docstring.

fig = plot_trend_curves(
    binned, languages=LANGUAGES, features=FEATURES,
    sentiment_min=s_min, sentiment_max=s_max,
    figsize=FIGSIZE, font_scale=FONT_SCALE,
    show_linear_fit=SHOW_LINEAR, show_ci_band=SHOW_CI_BAND,
    panel_hspace=PANEL_HSPACE, panel_wspace=PANEL_WSPACE,
    title=TITLE,
)
save_fig(fig, SAVE_PATH, dpi=SAVE_DPI)
plt.show()

# %% [markdown]
# ## Figure 3 — Global Trend Curve

# %% Fig 3 — Global trend
# ── Editorial variables ──────────────────────────────────────────────
FIGSIZE          = (10, 4)
FONT_SCALE       = 1.0
SHOW_SPLIT_LINE  = True        # purple dashed line at split_point
SHOW_MIN_MARKER  = True        # dot at curve minimum
TITLE            = "Global Trend — All Features and Languages"
YLABEL           = "Normalised feature value (avg.)"
XLABEL           = "Sentiment"
SAVE_PATH        = rdir / "figures/fig3_global.png"
SAVE_DPI         = 300
# ─────────────────────────────────────────────────────────────────────
# Available kwargs: figsize, font_scale, show_split_line,
#   show_minimum_marker, title, ylabel, xlabel
# See utils/plotting.py → plot_global_trend() for full docstring.

fig = plot_global_trend(
    gt_series, split_point=split_point,
    sentiment_min=s_min, sentiment_max=s_max,
    figsize=FIGSIZE, font_scale=FONT_SCALE,
    show_split_line=SHOW_SPLIT_LINE,
    show_minimum_marker=SHOW_MIN_MARKER,
    title=TITLE, ylabel=YLABEL, xlabel=XLABEL,
)
save_fig(fig, SAVE_PATH, dpi=SAVE_DPI)
plt.show()

# %% [markdown]
# ## Figure 4 — Diagnostic Split Analysis

# %% Fig 4 — Diagnostic split
# ── Editorial variables ──────────────────────────────────────────────
LANGUAGES      = langs
FEATURES       = feats
FIGSIZE        = None
FONT_SCALE     = 1.0
SHOW_SPLIT     = True          # purple split line
SHOW_NEG_FIT   = True          # red dashed on negative side
SHOW_POS_FIT   = True          # green dashed on positive side
SHOW_INFO_BOX  = True          # ✓/+/× symbol in corner of each panel
PANEL_HSPACE   = 0.35
PANEL_WSPACE   = 0.25
TITLE          = "Diagnostic Split Analysis (H3)"
SAVE_PATH      = rdir / "figures/fig4_diagnostic.png"
SAVE_DPI       = 300
# ─────────────────────────────────────────────────────────────────────
# Available kwargs: figsize, font_scale, show_split_line, show_neg_fit,
#   show_pos_fit, show_info_box, panel_hspace, panel_wspace, title
# Check symbols: ✓ = strong arousal pattern, + = partial, × = none
# See utils/plotting.py → plot_diagnostic_split() for full docstring.

fig = plot_diagnostic_split(
    binned, h3_raw, split_point,
    languages=LANGUAGES, features=FEATURES,
    sentiment_min=s_min, sentiment_max=s_max,
    figsize=FIGSIZE, font_scale=FONT_SCALE,
    show_split_line=SHOW_SPLIT,
    show_neg_fit=SHOW_NEG_FIT, show_pos_fit=SHOW_POS_FIT,
    show_info_box=SHOW_INFO_BOX,
    panel_hspace=PANEL_HSPACE, panel_wspace=PANEL_WSPACE,
    title=TITLE,
)
save_fig(fig, SAVE_PATH, dpi=SAVE_DPI)
plt.show()

# %% [markdown]
# ## Mirrored Dataset PNG (exploratory — not in paper by default)

# %% Mirrored PNG
# ── Toggle ──────────────────────────────────────────────────────────
GENERATE_MIRRORED = False      # set True to generate
# ─────────────────────────────────────────────────────────────────────
if GENERATE_MIRRORED:
    # Mirror each binned curve around its midpoint and overlay
    fig, axes = plt.subplots(len(feats), len(langs),
                              figsize=(18, len(feats) * 3.2), squeeze=False)
    import numpy as np
    x = np.linspace(s_min, s_max, cfg["analysis"]["n_bins"])
    for r, feat in enumerate(feats):
        for c, lang in enumerate(langs):
            ax = axes[r][c]
            curve = binned.get((lang, feat))
            if curve is not None:
                y = curve.values
                y_mirror = y[::-1]
                ax.plot(x, y, color=PALETTE["bin_curve"], lw=1.8, label="Original")
                ax.plot(x, y_mirror, color=PALETTE["linear_fit"],
                        lw=1.2, ls="--", label="Mirrored")
            if r == 0: ax.set_title(lang)
            if c == 0: ax.set_ylabel(feat)
    axes[0][0].legend(fontsize=8)
    plt.suptitle("Mirrored dataset check (exploratory)", fontsize=12)
    plt.tight_layout()
    save_fig(fig, rdir / "figures/fig_mirrored_exploratory.png", dpi=200)
    plt.show()
