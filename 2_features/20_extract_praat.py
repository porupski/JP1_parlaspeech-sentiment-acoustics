#!/usr/bin/env python3
# ============================================================
# Script:  20_extract_praat.py
# Release: 1.0
# Version: v1.00
# Purpose: Extract F0 and intensity per utterance via Praat (parselmouth).
#          Method: word-level median → utterance-level median (paper §3.3).
#
# Input:   {intermediate_dir}/{lang}_filtered.jsonl
# Output:  {intermediate_dir}/{lang}_praat.tsv
#          Columns: utterance_id, speaker_id, session_id, language,
#                   sentiment_score, sentiment_label, n_words,
#                   f0_raw, intensity_raw
#
# NOTE: Audio paths must be set in config.json → paths.audio_root.{lang}
# NOTE: Gender-specific pitch ranges applied if 'gender' field present in JSONL.
# ============================================================

import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir, get_audio_root
from utils.data_utils import load_jsonl, write_tsv
from utils.extraction import extract_praat_utterance


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers for audio processing")
    return p.parse_args()


def get_pitch_range(gender: str, cfg: dict) -> tuple[float, float]:
    pr = cfg["praat"]
    g = (gender or "").lower()
    if g in ("m", "male"):
        return pr["pitch_floor_male"], pr["pitch_ceiling_male"]
    elif g in ("f", "female"):
        return pr["pitch_floor_female"], pr["pitch_ceiling_female"]
    return pr["pitch_floor_default"], pr["pitch_ceiling_default"]


def process_record(rec: dict, audio_root: Path, cfg: dict) -> dict:
    audio_rel = rec.get("audio")
    if audio_rel is None:
        return {"utterance_id": rec["utterance_id"], "f0_raw": None, "intensity_raw": None}

    audio_path = audio_root / audio_rel
    if not audio_path.exists():
        return {"utterance_id": rec["utterance_id"], "f0_raw": None, "intensity_raw": None}

    pitch_floor, pitch_ceiling = get_pitch_range(rec.get("gender"), cfg)
    feats = extract_praat_utterance(
        audio_path=audio_path,
        words_align=rec.get("words_align", []),
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
        min_intensity_db=cfg["praat"]["intensity_min_db"],
    )
    return {"utterance_id": rec["utterance_id"], **feats}


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
        audio_root = get_audio_root(cfg, lang)
        print(f"\n[{lang}] Extracting Praat features from {len(records):,} utterances ...")
        print(f"  Audio root: {audio_root}")

        rows = []
        for rec in tqdm(records, desc=lang):
            meta = {
                "utterance_id": rec["utterance_id"],
                "speaker_id": rec["speaker_id"],
                "session_id": rec.get("session_id"),
                "language": lang,
                "sentiment_score": rec["sentiment_score"],
                "sentiment_label": rec["sentiment_label"],
                "n_words": rec["n_words"],
            }
            feats = process_record(rec, audio_root, cfg)
            rows.append({**meta, **feats})

        df = pd.DataFrame(rows)
        n_valid_f0 = df["f0_raw"].notna().sum()
        n_valid_int = df["intensity_raw"].notna().sum()
        print(f"  F0 valid: {n_valid_f0:,}/{len(df):,} "
              f"({100*n_valid_f0/len(df):.1f}%)")
        print(f"  Intensity valid: {n_valid_int:,}/{len(df):,} "
              f"({100*n_valid_int/len(df):.1f}%)")

        out = idir / f"{lang}_praat.tsv"
        write_tsv(df, out)
        print(f"  Written → {out}")


if __name__ == "__main__":
    main()
