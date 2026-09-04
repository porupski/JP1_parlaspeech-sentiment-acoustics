#!/usr/bin/env python3
# ============================================================
# Script:  11_join.py
# Release: 1.0
# Version: v1.01
# Purpose: Join feature TSVs into one combined feature TSV per language.
#          Metadata comes from filtered JSONL, not from praat.tsv,
#          so this works whether or not praat/opensmile stages were run.
#
# Input:   {intermediate_dir}/{lang}_filtered.jsonl  (required — metadata)
#          {intermediate_dir}/{lang}_speechrate.tsv   (required)
#          {intermediate_dir}/{lang}_praat.tsv        (optional)
#          {intermediate_dir}/{lang}_opensmile.tsv    (optional, if enable_opensmile)
# Output:  {intermediate_dir}/{lang}_features.tsv
# ============================================================

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir
from utils.data_utils import load_jsonl, write_tsv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    return p.parse_args()


def _merge_features(base: pd.DataFrame, feat_path: Path, label: str) -> pd.DataFrame:
    """Merge feature TSV onto base, skipping columns already present."""
    feat = pd.read_csv(feat_path, sep="\t")
    new_cols = ["utterance_id"] + [c for c in feat.columns
                                   if c != "utterance_id" and c not in base.columns]
    return base.merge(feat[new_cols], on="utterance_id", how="left")


def join_language(lang: str, idir: Path, cfg: dict) -> pd.DataFrame:
    filtered_path = idir / f"{lang}_filtered.jsonl"
    sr_path = idir / f"{lang}_speechrate.tsv"
    praat_path = idir / f"{lang}_praat.tsv"
    osmile_path = idir / f"{lang}_opensmile.tsv"

    if not filtered_path.exists():
        raise FileNotFoundError(f"Missing: {filtered_path}. Run 10_filter.py first.")
    if not sr_path.exists():
        raise FileNotFoundError(f"Missing: {sr_path}. Run 21_extract_speechrate.py first.")

    # Metadata base from filtered JSONL
    records = load_jsonl(filtered_path)
    df = pd.DataFrame([{
        "utterance_id": r["utterance_id"],
        "speaker_id": r["speaker_id"],
        "session_id": r.get("session_id"),
        "language": lang,
        "sentiment_score": r["sentiment_score"],
        "sentiment_label": r["sentiment_label"],
        "n_words": r["n_words"],
        "gender": r.get("gender"),
    } for r in records])

    # Speechrate (required)
    df = _merge_features(df, sr_path, "speechrate")

    # Praat (optional — skipped if praat stage was not run)
    if praat_path.exists():
        df = _merge_features(df, praat_path, "praat")
    else:
        print(f"  [INFO] {praat_path} not found — f0/intensity columns absent.")

    # OpenSMILE (optional)
    if cfg["analysis"]["enable_opensmile"]:
        if osmile_path.exists():
            df = _merge_features(df, osmile_path, "opensmile")
        else:
            print(f"  [WARN] OpenSMILE enabled but {osmile_path} not found — skipping.")

    return df


def main():
    args = parse_args()
    cfg = load_config(args.config)
    langs = args.langs or cfg["languages"]
    idir = get_intermediate_dir(cfg)

    for lang in langs:
        print(f"\n[{lang}] Joining features ...")
        df = join_language(lang, idir, cfg)
        out = idir / f"{lang}_features.tsv"
        write_tsv(df, out)
        print(f"[{lang}] {len(df):,} utterances × {len(df.columns)} columns → {out}")


if __name__ == "__main__":
    main()
