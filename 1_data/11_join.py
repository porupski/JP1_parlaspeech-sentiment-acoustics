#!/usr/bin/env python3
# ============================================================
# Script:  11_join.py
# Release: 1.0
# Version: v1.00
# Purpose: Join Praat + speech-rate + OpenSMILE feature TSVs into one
#          combined feature TSV per language. Add normalized columns.
#
# Input:   {intermediate_dir}/{lang}_praat.tsv
#          {intermediate_dir}/{lang}_speechrate.tsv
#          {intermediate_dir}/{lang}_opensmile.tsv  (if enable_opensmile)
# Output:  {intermediate_dir}/{lang}_features.tsv
# ============================================================

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir
from utils.data_utils import write_tsv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    return p.parse_args()


def join_language(lang: str, idir: Path, cfg: dict) -> pd.DataFrame:
    key_col = "utterance_id"

    praat_path = idir / f"{lang}_praat.tsv"
    sr_path = idir / f"{lang}_speechrate.tsv"
    osmile_path = idir / f"{lang}_opensmile.tsv"

    if not praat_path.exists():
        raise FileNotFoundError(f"Missing: {praat_path}. Run 20_extract_praat.py first.")
    if not sr_path.exists():
        raise FileNotFoundError(f"Missing: {sr_path}. Run 21_extract_speechrate.py first.")

    df = pd.read_csv(praat_path, sep="\t")
    sr = pd.read_csv(sr_path, sep="\t")
    df = df.merge(sr, on=key_col, how="inner", suffixes=("", "_sr"))

    if cfg["analysis"]["enable_opensmile"] and osmile_path.exists():
        osmile = pd.read_csv(osmile_path, sep="\t")
        df = df.merge(osmile, on=key_col, how="left", suffixes=("", "_osmile"))
    elif cfg["analysis"]["enable_opensmile"]:
        print(f"  [WARN] OpenSMILE enabled but {osmile_path} not found — skipping.")

    df["language"] = lang
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
