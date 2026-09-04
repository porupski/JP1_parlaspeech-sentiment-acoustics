#!/usr/bin/env python3
# ============================================================
# Script:  22_extract_opensmile.py
# Release: 1.0
# Version: v1.00
# Purpose: Extract OpenSMILE GeMAPS features per utterance (appendix track).
#          Disabled by default; enable via config enable_opensmile = true.
#
# Input:   {intermediate_dir}/{lang}_filtered.jsonl
# Output:  {intermediate_dir}/{lang}_opensmile.tsv
# ============================================================

import sys
import argparse
import numpy as np
import pandas as pd
import opensmile
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir, get_audio_root
from utils.data_utils import load_jsonl, write_tsv


SMILE = None  # lazy init


def get_smile():
    global SMILE
    if SMILE is None:
        SMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return SMILE


def extract_opensmile_utterance(audio_path: Path,
                                 start: float, end: float) -> dict:
    """Extract GeMAPS functionals for one utterance segment."""
    smile = get_smile()
    try:
        feats = smile.process_file(str(audio_path), start=start, end=end)
        # feats is a DataFrame with one row and GeMAPS columns
        row = feats.iloc[0].to_dict()
        # prefix all keys
        return {f"osmile_{k}": v for k, v in row.items()}
    except Exception as e:
        return {}


def utterance_time_bounds(words_align: list[dict]) -> tuple[float, float]:
    if not words_align:
        return (0.0, 0.0)
    return words_align[0]["start"], words_align[-1]["end"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if not cfg["analysis"]["enable_opensmile"]:
        print("OpenSMILE disabled in config (enable_opensmile = false). Exiting.")
        return

    langs = args.langs or cfg["languages"]
    idir = get_intermediate_dir(cfg)

    for lang in langs:
        in_path = idir / f"{lang}_filtered.jsonl"
        if not in_path.exists():
            print(f"[{lang}] {in_path} not found. Run 10_filter.py first.")
            continue

        records = load_jsonl(in_path)
        audio_root = get_audio_root(cfg, lang)
        print(f"\n[{lang}] Extracting OpenSMILE GeMAPS from {len(records):,} utterances ...")

        rows = []
        for rec in tqdm(records, desc=lang):
            audio_rel = rec.get("audio")
            if audio_rel is None:
                rows.append({"utterance_id": rec["utterance_id"]})
                continue
            audio_path = audio_root / audio_rel
            if not audio_path.exists():
                rows.append({"utterance_id": rec["utterance_id"]})
                continue
            start, end = utterance_time_bounds(rec.get("words_align", []))
            feats = extract_opensmile_utterance(audio_path, start, end)
            rows.append({"utterance_id": rec["utterance_id"], **feats})

        df = pd.DataFrame(rows)
        out = idir / f"{lang}_opensmile.tsv"
        write_tsv(df, out)
        print(f"  Written → {out} ({len(df.columns)} cols)")


if __name__ == "__main__":
    main()
