# ============================================================
# Script:  plotting.py
# Release: 1.0
# Version: v1.00
# Purpose: All figure generation. One palette, all editorial switches as kwargs.
#          Import this; do not copy it.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Shared palette — one place, applied everywhere
# ---------------------------------------------------------------------------
PALETTE = {
    # per-language colors
    "HR": "#E63946",
    "CZ": "#2A9D8F",
    "PL": "#E9C46A",
    "RS": "#264653",
    "SI": "#A8DADC",
    # curve and fit line colors (match paper captions)
    "bin_curve": "#F4A261",      # orange — 60-bin speaker-averaged curve
    "linear_fit": "#457B9D",     # blue dashed — full-range linear fit
    "split_line": "#9B5DE5",     # purple dashed — split point
    "neg_fit": "#E63946",        # red dashed — negative-side linear fit
    "pos_fit": "#2DC653",        # green dashed — positive-side linear fit
    "ci_band": "#F4A26140",      # orange semi-transparent — CI shading
    # bar chart
    "bar_sig": "#264653",
    "bar_ns": "#ADB5BD",
    "chance_line": "#E63946",
}

LANG_ORDER = ["HR", "CZ", "PL", "RS", "SI"]
FEATURE_ORDER = ["f0_raw", "intensity_norm", "speechrate_wps"]
FEATURE_LABELS = {
    "f0_raw": "F0 (pitch, Hz)",
    "f0_norm": "F0 (norm.)",
    "intensity_raw": "Intensity (dB)",
    "intensity_norm": "Intensity (dB, norm.)",
    "speechrate_wps": "Speech rate (words/s)",
    "speechrate_sps": "Speech rate (syl/s)",
}


def _apply_font_scale(ax, font_scale: float = 1.0) -> None:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(item.get_fontsize() * font_scale)


# ---------------------------------------------------------------------------
# Fig 1 — Concordance probability bar chart
# ---------------------------------------------------------------------------

def plot_concordance_bars(
    results: dict,
    languages: list[str] = None,
    features: list[str] = None,
    *,
    figsize: tuple = (12, 5),
    font_scale: float = 1.0,
    show_chance_line: bool = True,
    chance_value: float = 0.5,
    bar_alpha: float = 0.85,
    title: str = "Concordance P(Neg > Pos) — H1",
    xlabel: str = "P(Negative > Positive)",
) -> plt.Figure:
    """
    Horizontal bar chart of concordance probabilities from H1.

    results: dict keyed by (language, feature) → dict with keys:
        'speaker_avg': {'concordance': float, 'p': float}
        'utterance_level': {'concordance': float, 'p': float}

    Kwargs (editorial controls):
        figsize          (width, height) in inches
        font_scale       global font size multiplier
        show_chance_line draw vertical line at chance_value
        chance_value     where to draw the chance line (default 0.5)
        bar_alpha        bar transparency
        title            figure title
        xlabel           x-axis label
    """
    if languages is None:
        languages = LANG_ORDER
    if features is None:
        features = FEATURE_ORDER

    fig, axes = plt.subplots(1, len(features), figsize=figsize, sharey=False)
    if len(features) == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        labels, sa_vals, ul_vals, sa_sig, ul_sig = [], [], [], [], []
        for lang in languages:
            key = (lang, feat)
            if key not in results:
                continue
            r = results[key]
            sa = r.get("speaker_avg", {})
            ul = r.get("utterance_level", {})
            labels.append(lang)
            sa_vals.append(sa.get("concordance", np.nan))
            ul_vals.append(ul.get("concordance", np.nan))
            sa_sig.append(sa.get("p", 1.0) < 0.05)
            ul_sig.append(ul.get("p", 1.0) < 0.05)

        y = np.arange(len(labels))
        h = 0.35
        for i, (v, sig) in enumerate(zip(sa_vals, sa_sig)):
            color = PALETTE["bar_sig"] if sig else PALETTE["bar_ns"]
            ax.barh(y[i] + h / 2, v, h, color=color, alpha=bar_alpha, label="Speaker avg" if i == 0 else "")
        for i, (v, sig) in enumerate(zip(ul_vals, ul_sig)):
            color = PALETTE["bar_sig"] if sig else PALETTE["bar_ns"]
            ax.barh(y[i] - h / 2, v, h, color=color, alpha=bar_alpha * 0.6, label="Utterance level" if i == 0 else "")

        if show_chance_line:
            ax.axvline(chance_value, color=PALETTE["chance_line"], lw=1.5, ls="--", alpha=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_title(FEATURE_LABELS.get(feat, feat))
        ax.set_xlim(0, 1)
        _apply_font_scale(ax, font_scale)

    fig.suptitle(title, fontsize=12 * font_scale)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 2 — 60-bin trend curves (features × languages grid)
# ---------------------------------------------------------------------------

def plot_trend_curves(
    binned_data: dict,
    languages: list[str] = None,
    features: list[str] = None,
    sentiment_min: float = 0.0,
    sentiment_max: float = 5.0,
    *,
    figsize: tuple = None,
    font_scale: float = 1.0,
    show_linear_fit: bool = True,
    show_ci_band: bool = False,
    ci_data: dict = None,
    panel_hspace: float = 0.35,
    panel_wspace: float = 0.25,
    title: str = "Sentiment–Acoustic Trends (H2)",
) -> plt.Figure:
    """
    Multi-panel grid: rows=features, cols=languages.
    binned_data: dict keyed by (language, feature) → pd.Series(index=bin, values=mean)

    Kwargs (editorial controls):
        figsize           (w, h) — auto if None (18 × n_features * 3.2)
        font_scale        global font size multiplier
        show_linear_fit   overlay full-range linear regression line
        show_ci_band      shade bootstrap CI around bin curve
        ci_data           dict (lang, feat) → (lower_series, upper_series) for CI band
        panel_hspace      vertical spacing between rows
        panel_wspace      horizontal spacing between columns
        title             figure suptitle
    """
    if languages is None:
        languages = LANG_ORDER
    if features is None:
        features = FEATURE_ORDER
    n_feat = len(features)
    n_lang = len(languages)

    if figsize is None:
        figsize = (18, n_feat * 3.2)

    fig, axes = plt.subplots(n_feat, n_lang, figsize=figsize,
                              squeeze=False, sharex=True)
    fig.subplots_adjust(hspace=panel_hspace, wspace=panel_wspace)

    x_vals = np.linspace(sentiment_min, sentiment_max, 60)

    for r, feat in enumerate(features):
        for c, lang in enumerate(languages):
            ax = axes[r][c]
            key = (lang, feat)
            curve = binned_data.get(key)

            if curve is not None and not curve.isna().all():
                y = curve.values
                ax.plot(x_vals[:len(y)], y, color=PALETTE["bin_curve"],
                        lw=1.8, label="Bin mean")

                if show_ci_band and ci_data and key in ci_data:
                    lo, hi = ci_data[key]
                    ax.fill_between(x_vals[:len(y)], lo, hi,
                                    color=PALETTE["bin_curve"], alpha=0.2)

                if show_linear_fit:
                    valid = ~np.isnan(y)
                    if valid.sum() > 2:
                        xv = x_vals[:len(y)][valid]
                        yv = y[valid]
                        sl, ic, *_ = stats.linregress(xv, yv)
                        ax.plot(xv, sl * xv + ic,
                                color=PALETTE["linear_fit"], lw=1.2, ls="--")

            if r == 0:
                ax.set_title(lang, fontsize=11 * font_scale)
            if c == 0:
                ax.set_ylabel(FEATURE_LABELS.get(feat, feat),
                              fontsize=9 * font_scale)
            if r == n_feat - 1:
                ax.set_xlabel("Sentiment", fontsize=9 * font_scale)
            ax.tick_params(labelsize=8 * font_scale)

    fig.suptitle(title, fontsize=12 * font_scale)
    return fig


# ---------------------------------------------------------------------------
# Fig 3 — Global trend curve
# ---------------------------------------------------------------------------

def plot_global_trend(
    global_curve: pd.Series,
    split_point: Optional[float] = None,
    sentiment_min: float = 0.0,
    sentiment_max: float = 5.0,
    *,
    figsize: tuple = (10, 4),
    font_scale: float = 1.0,
    show_split_line: bool = True,
    show_minimum_marker: bool = True,
    title: str = "Global Trend — All Features and Languages",
    ylabel: str = "Normalised feature value (avg.)",
    xlabel: str = "Sentiment",
) -> plt.Figure:
    """
    Single panel: normalised global average across all features and languages.

    Kwargs (editorial controls):
        figsize              (w, h)
        font_scale           global font size multiplier
        show_split_line      draw vertical dashed line at split_point
        show_minimum_marker  mark the minimum with a dot
        title / ylabel / xlabel  labels
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.linspace(sentiment_min, sentiment_max, len(global_curve))
    y = global_curve.values

    ax.plot(x, y, color=PALETTE["bin_curve"], lw=2.0)

    if show_minimum_marker:
        min_idx = np.nanargmin(y)
        ax.scatter([x[min_idx]], [y[min_idx]], color=PALETTE["split_line"],
                   s=60, zorder=5, label=f"Min at {x[min_idx]:.2f}")

    if show_split_line and split_point is not None:
        ax.axvline(split_point, color=PALETTE["split_line"], lw=1.5, ls="--",
                   label=f"Split = {split_point:.2f}")
        ax.legend(fontsize=9 * font_scale)

    ax.set_xlabel(xlabel, fontsize=10 * font_scale)
    ax.set_ylabel(ylabel, fontsize=10 * font_scale)
    ax.set_title(title, fontsize=12 * font_scale)
    ax.tick_params(labelsize=9 * font_scale)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 4 — Diagnostic split plots
# ---------------------------------------------------------------------------

def plot_diagnostic_split(
    binned_data: dict,
    split_results: dict,
    split_point: float,
    languages: list[str] = None,
    features: list[str] = None,
    sentiment_min: float = 0.0,
    sentiment_max: float = 5.0,
    *,
    figsize: tuple = None,
    font_scale: float = 1.0,
    show_split_line: bool = True,
    show_neg_fit: bool = True,
    show_pos_fit: bool = True,
    show_info_box: bool = True,
    panel_hspace: float = 0.35,
    panel_wspace: float = 0.25,
    title: str = "Diagnostic Split Analysis (H3)",
) -> plt.Figure:
    """
    Same grid as Fig 2, with split line and per-side linear fits.
    split_results: dict (lang, feat) → {'neg': h3_result_dict, 'pos': h3_result_dict, 'check': str}

    Kwargs (editorial controls):
        figsize            auto if None
        font_scale         global font size multiplier
        show_split_line    draw purple vertical dashed line
        show_neg_fit       draw red dashed line on negative side
        show_pos_fit       draw green dashed line on positive side
        show_info_box      overlay text box with tau, p, check symbol
        panel_hspace/wspace spacing
        title              figure suptitle

    Check symbols in info box: ✓ strong arousal, + partial, × none
    """
    if languages is None:
        languages = LANG_ORDER
    if features is None:
        features = FEATURE_ORDER
    n_feat = len(features)
    n_lang = len(languages)

    if figsize is None:
        figsize = (18, n_feat * 3.2)

    fig, axes = plt.subplots(n_feat, n_lang, figsize=figsize,
                              squeeze=False, sharex=True)
    fig.subplots_adjust(hspace=panel_hspace, wspace=panel_wspace)

    x_all = np.linspace(sentiment_min, sentiment_max, 60)
    split_idx = int((split_point - sentiment_min) / (sentiment_max - sentiment_min) * 60)

    for r, feat in enumerate(features):
        for c, lang in enumerate(languages):
            ax = axes[r][c]
            key = (lang, feat)
            curve = binned_data.get(key)

            if curve is not None and not curve.isna().all():
                y = curve.values
                ax.plot(x_all[:len(y)], y, color=PALETTE["bin_curve"], lw=1.8)

                if show_neg_fit and key in split_results:
                    neg_r = split_results[key].get("neg", {})
                    sl = neg_r.get("linear_slope")
                    if sl is not None and not np.isnan(sl):
                        xn = x_all[:split_idx + 1]
                        yn_valid = y[:split_idx + 1]
                        valid = ~np.isnan(yn_valid)
                        if valid.sum() > 2:
                            sl2, ic2, *_ = stats.linregress(xn[valid], yn_valid[valid])
                            ax.plot(xn, sl2 * xn + ic2,
                                    color=PALETTE["neg_fit"], lw=1.2, ls="--")

                if show_pos_fit and key in split_results:
                    pos_r = split_results[key].get("pos", {})
                    sl = pos_r.get("linear_slope")
                    if sl is not None and not np.isnan(sl):
                        xp = x_all[split_idx:]
                        yp_valid = y[split_idx:]
                        valid = ~np.isnan(yp_valid)
                        if valid.sum() > 2:
                            sl2, ic2, *_ = stats.linregress(xp[valid], yp_valid[valid])
                            ax.plot(xp, sl2 * xp + ic2,
                                    color=PALETTE["pos_fit"], lw=1.2, ls="--")

            if show_split_line:
                ax.axvline(split_point, color=PALETTE["split_line"],
                           lw=1.2, ls="--", alpha=0.8)

            if show_info_box and key in split_results:
                check = split_results[key].get("check", "?")
                sym = {"strong": "✓", "partial": "+", "none": "×"}.get(check, "?")
                ax.text(0.97, 0.97, sym, transform=ax.transAxes,
                        ha="right", va="top", fontsize=12 * font_scale,
                        color="green" if check == "strong" else
                              "orange" if check == "partial" else "red")

            if r == 0:
                ax.set_title(lang, fontsize=11 * font_scale)
            if c == 0:
                ax.set_ylabel(FEATURE_LABELS.get(feat, feat), fontsize=9 * font_scale)
            if r == n_feat - 1:
                ax.set_xlabel("Sentiment", fontsize=9 * font_scale)
            ax.tick_params(labelsize=8 * font_scale)

    fig.suptitle(title, fontsize=12 * font_scale)
    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
