#!/usr/bin/env python3
# ============================================================
# Script:  21_extract_speechrate.py
# Release: 1.0
# Version: v1.01
# Purpose: Compute speech rate and pause features from word-level timing.
#          No audio needed — uses alignment data from filtered JSONL.
#
# Input:   {intermediate_dir}/{lang}_filtered.jsonl
# Output:  {intermediate_dir}/{lang}_speechrate.tsv
#
# v1.01: Added pause breakdown from v4 silent_pauses + filled_pauses tiers.
#        pause_ratio renamed to pause_ratio_all (old formula).
#        pause_ratio_silent (confirmed silent pause duration / total) is new main metric.
# ============================================================

import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir
from utils.data_utils import load_jsonl, write_tsv
from utils.extraction import extract_speechrate_utterance


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    langs = args.langs or cfg["languages"]
    idir = get_intermediate_dir(cfg)

    for lang in langs:
        in_path = idir / f"{lang}_filtered.jsonl"
        if not in_path.exists():
            print(f"[{lang}] {in_path} not found. Run 10_filter.py first.")
            continue

        records = load_jsonl(in_path)
        print(f"\n[{lang}] Computing speech rate for {len(records):,} utterances ...")

        if not records:
            print(f"  [SKIP] No records.")
            continue

        rows = []
        for rec in tqdm(records, desc=lang):
            feats = extract_speechrate_utterance(
                words_align=rec.get("words_align", []),
                lang=lang,
                silent_pauses=rec.get("silent_pauses"),
                filled_pauses=rec.get("filled_pauses"),
            )
            rows.append({"utterance_id": rec["utterance_id"], **feats})

        df = pd.DataFrame(rows)
        n = len(df)
        print(f"  speechrate_wps valid:     {df['speechrate_wps'].notna().sum():,}/{n:,}")
        print(f"  pause_ratio_silent valid: {df['pause_ratio_silent'].notna().sum():,}/{n:,}")
        print(f"  n_silent_pauses mean:     {df['n_silent_pauses'].mean():.2f}")
        print(f"  n_filled_pauses mean:     {df['n_filled_pauses'].mean():.2f}")

        out = idir / f"{lang}_speechrate.tsv"
        write_tsv(df, out)
        print(f"  Written → {out}")


if __name__ == "__main__":
    main()
