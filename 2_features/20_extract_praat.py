#!/usr/bin/env python3
# ============================================================
# Script:  20_extract_praat.py
# Release: 1.0
# Version: v1.01
# Purpose: Extract F0 and intensity per utterance via Praat (parselmouth).
#          Method: word-level median → utterance-level median (paper §3.3).
#
# Input:   {intermediate_dir}/{lang}_filtered.jsonl
# Output:  {intermediate_dir}/{lang}_praat.tsv
#          Columns: utterance_id, speaker_id, session_id, language,
#                   sentiment_score, sentiment_label, n_words,
#                   f0_raw, intensity_raw
#
# NOTE: Audio paths set in config.json → paths.audio_root.{lang}.
#       Script builds a filename index to handle nested part-folder layouts.
# NOTE: Gender-specific pitch ranges applied if 'gender' field present in JSONL.
# ============================================================

import os
import sys
import signal
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import (
    load_config, get_intermediate_dir, get_audio_root,
    build_audio_index, resolve_audio_path,
)
from utils.data_utils import load_jsonl, write_tsv
from utils.extraction import extract_praat_utterance


# Worker-process globals set by _init_worker
_G_INDEX: dict = {}
_G_ROOT: "Path | None" = None
_G_CFG: dict = {}


def _init_worker(index: dict, audio_root_str: str, cfg: dict) -> None:
    global _G_INDEX, _G_ROOT, _G_CFG
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 19)
    except OSError:
        pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _G_INDEX = index
    _G_ROOT = Path(audio_root_str)
    _G_CFG = cfg


def _get_pitch_range(gender: str, cfg: dict) -> tuple:
    pr = cfg["praat"]
    g = (gender or "").lower()
    if g in ("m", "male"):
        return pr["pitch_floor_male"], pr["pitch_ceiling_male"]
    if g in ("f", "female"):
        return pr["pitch_floor_female"], pr["pitch_ceiling_female"]
    return pr["pitch_floor_default"], pr["pitch_ceiling_default"]


def _process_record_worker(rec: dict) -> dict:
    uid = rec["utterance_id"]
    null = {"utterance_id": uid, "f0_raw": None, "intensity_raw": None}
    audio_rel = rec.get("audio")
    if not audio_rel:
        return null
    audio_path = resolve_audio_path(audio_rel, _G_ROOT, _G_INDEX)
    if audio_path is None:
        return null
    pitch_floor, pitch_ceiling = _get_pitch_range(rec.get("gender"), _G_CFG)
    try:
        feats = extract_praat_utterance(
            audio_path=audio_path,
            words_align=rec.get("words_align", []),
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
            min_intensity_db=_G_CFG["praat"]["intensity_min_db"],
        )
    except Exception:
        return null
    return {"utterance_id": uid, **feats}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel worker processes for audio extraction")
    return p.parse_args()


def main():
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 19)
    except OSError:
        pass
    args = parse_args()
    cfg = load_config(args.config)
    langs = args.langs or cfg["languages"]
    idir = get_intermediate_dir(cfg)

    for lang in langs:
        out = idir / f"{lang}_praat.tsv"
        if out.exists():
            print(f"[{lang}] {out} already exists — skipping. Delete to rerun.")
            continue

        in_path = idir / f"{lang}_filtered.jsonl"
        if not in_path.exists():
            print(f"[{lang}] {in_path} not found. Run 10_filter.py first.")
            continue

        records = load_jsonl(in_path)
        audio_root = get_audio_root(cfg, lang)
        print(f"\n[{lang}] Building audio index from {audio_root} ...")
        index = build_audio_index(audio_root)
        print(f"[{lang}] Index: {len(index):,} audio files")
        print(f"[{lang}] Extracting Praat features from {len(records):,} utterances "
              f"({args.workers} workers) ...")

        if args.workers <= 1:
            # Single-process: set globals directly
            global _G_INDEX, _G_ROOT, _G_CFG
            _G_INDEX = index
            _G_ROOT = audio_root
            _G_CFG = cfg
            feats_list = [_process_record_worker(r) for r in tqdm(records, desc=lang)]
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(index, str(audio_root), cfg),
            ) as pool:
                feats_list = list(tqdm(
                    pool.map(_process_record_worker, records, chunksize=50),
                    total=len(records),
                    desc=lang,
                ))

        meta_by_uid = {r["utterance_id"]: r for r in records}
        rows = []
        for feats in feats_list:
            uid = feats["utterance_id"]
            rec = meta_by_uid[uid]
            rows.append({
                "utterance_id": uid,
                "speaker_id": rec["speaker_id"],
                "session_id": rec.get("session_id"),
                "language": lang,
                "sentiment_score": rec["sentiment_score"],
                "sentiment_label": rec["sentiment_label"],
                "n_words": rec["n_words"],
                "f0_raw": feats["f0_raw"],
                "intensity_raw": feats["intensity_raw"],
            })

        df = pd.DataFrame(rows)
        n_f0 = df["f0_raw"].notna().sum()
        n_int = df["intensity_raw"].notna().sum()
        n = len(df)
        print(f"  F0 valid:        {n_f0:,}/{n:,} ({100*n_f0/n:.1f}%)")
        print(f"  Intensity valid: {n_int:,}/{n:,} ({100*n_int/n:.1f}%)")

        out = idir / f"{lang}_praat.tsv"
        write_tsv(df, out)
        print(f"  Written → {out}")


if __name__ == "__main__":
    main()
