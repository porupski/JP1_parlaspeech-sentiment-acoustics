# ============================================================
# Script:  explore_utils.py
# Release: 1.0
# Version: v1.00
# Purpose: Helpers for 50_explore.py — fig saving with descriptive names,
#          verbose logging, per-speaker z-scoring. Kept small on purpose;
#          import into the notebook, don't reach into it.
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FigSaver:
    """Descriptive fig naming with a monotonic counter for uniqueness.

    Names are formed as `NNN_<name>.png` where NNN is a zero-padded serial
    (survives duplicate names within a run). In display mode, calls plt.show().
    """

    def __init__(self, out_dir: Path | None, headless: bool, dpi: int = 120):
        self.out_dir = Path(out_dir) if out_dir else None
        self.headless = bool(headless)
        self.dpi = dpi
        self._n = 0
        if self.headless and self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, name: str, close: bool = True) -> Path | None:
        self._n += 1
        if self.headless and self.out_dir is not None:
            safe = _slug(name)
            fp = self.out_dir / f"{self._n:03d}_{safe}.png"
            plt.gcf().savefig(fp, dpi=self.dpi, bbox_inches="tight")
            if close:
                plt.close("all")
            return fp
        plt.show()
        if close:
            plt.close("all")
        return None


def _slug(s: str) -> str:
    """Lowercase, alnum + underscore, safe for filenames."""
    out = []
    for c in s:
        out.append(c if (c.isalnum() or c in "._-") else "_")
    slug = "".join(out).strip("._-")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "fig"


def logprint(label: str, value: Any = None, fmt: str | None = None, indent: int = 0) -> None:
    """Print a labelled value in a consistent, log-friendly format.
    Every numeric metric that appears on a plot should also go through this,
    so headless logs contain the full story.
    """
    pad = "  " * indent
    if value is None:
        print(f"{pad}{label}")
        return
    if fmt:
        print(f"{pad}{label}: {value:{fmt}}")
    else:
        print(f"{pad}{label}: {value}")


def zscore_per_speaker(df: pd.DataFrame, feat_col: str,
                        speaker_col: str = "speaker_id") -> pd.Series:
    """Return per-speaker z-scored values for a feature.
    Removes gender-driven baseline effects for F0 (SD-08 rule).
    """
    grp = df.groupby(speaker_col)[feat_col]
    return (df[feat_col] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)


def binned_speaker_mean(df: pd.DataFrame, feat_col: str, n_bins: int,
                          speaker_col: str = "speaker_id",
                          score_col: str = "sentiment_score",
                          score_range: tuple[float, float] = (0.0, 5.0)) -> np.ndarray:
    """Per-speaker bin means → equal-speaker mean per bin.
    Returns array of length n_bins (may contain NaN).
    """
    if feat_col not in df.columns or df[feat_col].isna().all():
        return np.full(n_bins, np.nan)
    sub = df.dropna(subset=[feat_col, score_col]).copy()
    if sub.empty:
        return np.full(n_bins, np.nan)
    edges = np.linspace(score_range[0], score_range[1], n_bins + 1)
    sub["_bin"] = pd.cut(sub[score_col], bins=edges, labels=False, include_lowest=True)
    per = sub.groupby([speaker_col, "_bin"])[feat_col].mean().reset_index()
    return per.groupby("_bin")[feat_col].mean().reindex(range(n_bins)).values.astype(float)


def clean_trend_plot(ax, x: np.ndarray, y: np.ndarray, title: str,
                      xlabel: str = "Sentiment", ylabel: str = "",
                      color: str = "#FFA500", lw: float = 3.0) -> dict:
    """Editorial style: single thick line, no overlays.
    Prints and returns {'slope','intercept','r','n_bins_valid'} for logging.
    """
    valid = ~np.isnan(y)
    n_v = int(valid.sum())
    stats = {"n_bins_valid": n_v, "slope": np.nan, "intercept": np.nan, "r": np.nan}

    if n_v >= 2:
        from scipy.stats import linregress
        lr = linregress(x[valid], y[valid])
        stats.update(slope=float(lr.slope), intercept=float(lr.intercept), r=float(lr.rvalue))
        ax.plot(x[valid], y[valid], color=color, lw=lw)
    else:
        ax.plot(x, y, color=color, lw=lw)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25)
    return stats


def spearman_report(x: pd.Series, y: pd.Series, label: str = "") -> dict:
    """Compute Spearman ρ + p on non-NA pairs; print and return dict."""
    from scipy.stats import spearmanr
    sub = pd.concat([x, y], axis=1).dropna()
    if len(sub) < 3:
        logprint(f"{label} Spearman", "n<3, skipped")
        return {"rho": np.nan, "p": np.nan, "n": len(sub)}
    rho, p = spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])
    print(f"  {label} Spearman ρ = {rho:+.4f}  p = {p:.3e}  n = {len(sub):,}")
    return {"rho": float(rho), "p": float(p), "n": int(len(sub))}
