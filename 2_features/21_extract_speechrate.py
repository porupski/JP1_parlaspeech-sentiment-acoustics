#!/usr/bin/env python3
# ============================================================
# Script:  21_extract_speechrate.py
# Release: 1.0
# Version: v1.00
# Purpose: Compute speech rate features from word-level timing in the
#          filtered JSONL (no audio needed).
#
# Input:   {intermediate_dir}/{lang}_filtered.jsonl
# Output:  {intermediate_dir}/{lang}_speechrate.tsv
#          Columns: utterance_id, n_words, n_syllables,
#                   duration_total, duration_speech, pause_ratio,
#                   speechrate_wps, speechrate_sps
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
            print(f"  [SKIP] No records after filter — skipping.")
            continue

        rows = []
        for rec in tqdm(records, desc=lang):
            feats = extract_speechrate_utterance(rec.get("words_align", []), lang)
            rows.append({"utterance_id": rec["utterance_id"], **feats})

        df = pd.DataFrame(rows)
        n_valid = df["speechrate_wps"].notna().sum() if "speechrate_wps" in df.columns else 0
        print(f"  speechrate_wps valid: {n_valid:,}/{len(df):,}")

        out = idir / f"{lang}_speechrate.tsv"
        write_tsv(df, out)
        print(f"  Written → {out}")


if __name__ == "__main__":
    main()
