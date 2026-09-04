# ============================================================
# Script:  data_utils.py
# Release: 1.0
# Version: v1.00
# Purpose: Data loading, filtering, and merging utilities
# ============================================================

import json
import numpy as np
import pandas as pd
from pathlib import Path


SENTIMENT_LABELS = [
    "Negative",
    "Mixed Negative",
    "Neutral Negative",
    "Neutral Positive",
    "Mixed Positive",
    "Positive",
]
SENTIMENT_LABEL_TO_IDX = {l: i for i, l in enumerate(SENTIMENT_LABELS)}


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_tsv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kwargs)


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def score_to_label(score: float) -> str:
    idx = min(max(round(score), 0), len(SENTIMENT_LABELS) - 1)
    return SENTIMENT_LABELS[idx]


def score_to_bin(score: float, n_bins: int = 60,
                 s_min: float = 0.0, s_max: float = 5.0) -> int:
    bin_width = (s_max - s_min) / n_bins
    return int(min((score - s_min) / bin_width, n_bins - 1))


def filter_word_count(df: pd.DataFrame, min_w: int = 10, max_w: int = 40,
                      col: str = "n_words") -> pd.DataFrame:
    return df[(df[col] >= min_w) & (df[col] <= max_w)].copy()


def filter_speakers_by_coverage(df: pd.DataFrame,
                                 min_per_label: int = 10,
                                 n_labels: int = 6,
                                 speaker_col: str = "speaker_id",
                                 label_col: str = "sentiment_label") -> pd.DataFrame:
    """Keep speakers who have >= min_per_label utterances in ALL n_labels categories."""
    counts = (
        df.groupby([speaker_col, label_col])
        .size()
        .unstack(fill_value=0)
    )
    # must have all n_labels columns present and all >= min_per_label
    if counts.shape[1] < n_labels:
        return df.iloc[0:0].copy()
    valid = counts[(counts >= min_per_label).all(axis=1)].index
    return df[df[speaker_col].isin(valid)].copy()


def get_extreme_pairs(df: pd.DataFrame,
                      speaker_col: str = "speaker_id",
                      label_col: str = "sentiment_label") -> pd.DataFrame:
    """Return rows labelled Negative or Positive only."""
    return df[df[label_col].isin(["Negative", "Positive"])].copy()


def compute_bins(df: pd.DataFrame, score_col: str = "sentiment_score",
                 n_bins: int = 60) -> pd.DataFrame:
    df = df.copy()
    df["bin"] = df[score_col].apply(score_to_bin, n_bins=n_bins)
    return df


def speaker_binned_means(df: pd.DataFrame,
                          feature_cols: list[str],
                          speaker_col: str = "speaker_id",
                          bin_col: str = "bin",
                          n_bins: int = 60) -> pd.DataFrame:
    """
    Per-speaker bin means, then mean across speakers (equal weight per speaker).
    Returns DataFrame with index=bin (0..n_bins-1), columns=feature_cols.
    """
    per_spk = df.groupby([speaker_col, bin_col])[feature_cols].mean()
    return per_spk.groupby(level=bin_col).mean().reindex(range(n_bins))


def speaker_binned_curves(df: pd.DataFrame,
                           feature_cols: list[str],
                           speaker_col: str = "speaker_id",
                           bin_col: str = "bin",
                           n_bins: int = 60) -> dict[str, pd.DataFrame]:
    """
    Return per-speaker bin-mean curves as dict {speaker_id: DataFrame(bin × features)}.
    """
    curves = {}
    for spk, grp in df.groupby(speaker_col):
        curve = grp.groupby(bin_col)[feature_cols].mean().reindex(range(n_bins))
        curves[spk] = curve
    return curves


def merge_feature_tsvs(paths: list[Path | str],
                        on: list[str] | None = None) -> pd.DataFrame:
    if on is None:
        on = ["utterance_id"]
    dfs = [pd.read_csv(p, sep="\t") for p in paths]
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=on, how="inner")
    return result
