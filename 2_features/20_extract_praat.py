#!/usr/bin/env python3
# ============================================================
# Script:  20_extract_praat.py
# Release: 1.0
# Version: v1.10
# Purpose: Extract acoustic features per utterance via Praat (parselmouth).
#          Method: utterance-level Praat objects sampled per word → word
#          median → utterance median (paper §3.3).
#
# Features extracted:
#   Scalar TSV — f0_raw, intensity_raw, f1_median, f2_median, f3_median,
#                hnr_median, jitter_local, jitter_rap, shimmer_local,
#                shimmer_local_db, shimmer_apq11, hnr_utt
#   Envelope NPZ (if praat.save_envelopes=true) — raw F0/intensity tracks
#                + per-word mean/median arrays for trajectory plotting
#
# Input:   {intermediate_dir}/{lang}_filtered.jsonl
# Output:  {intermediate_dir}/{lang}_praat.tsv
#          {intermediate_dir}/{lang}_praat_envelopes.npz  (if save_envelopes)
#
# v1.10: silent_pauses used to mask PointProcess; formants/HNR added;
#        envelope NPZ output; save_envelopes config flag.
# ============================================================

import os
import sys
import signal
import argparse
import numpy as np
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
    pr = _G_CFG["praat"]
    null = {"utterance_id": uid,
            "f0_raw": None, "intensity_raw": None,
            "f1_median": None, "f2_median": None, "f3_median": None,
            "hnr_median": None,
            "jitter_local": None, "jitter_rap": None,
            "shimmer_local": None, "shimmer_local_db": None, "shimmer_apq11": None,
            "hnr_utt": None,
            "_envelopes": None}

    audio_rel = rec.get("audio")
    if not audio_rel:
        return null
    audio_path = resolve_audio_path(audio_rel, _G_ROOT, _G_INDEX)
    if audio_path is None:
        return null

    pitch_floor, pitch_ceiling = _get_pitch_range(rec.get("gender"), _G_CFG)
    save_env = pr.get("save_envelopes", False)

    try:
        feats = extract_praat_utterance(
            audio_path=audio_path,
            words_align=rec.get("words_align", []),
            silent_pauses=rec.get("silent_pauses"),
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
            min_intensity_db=pr["intensity_min_db"],
            formant_n_samples=pr.get("formant_n_samples", 5),
            formant_max_hz=pr.get("formant_max_hz", 5500.0),
            save_envelopes=save_env,
        )
    except Exception:
        return null

    return {"utterance_id": uid, **feats}


def _save_envelopes_npz(path: Path, uid_envelope_pairs: list) -> None:
    """
    Save per-utterance envelope data to a single compressed npz.

    Load example:
        data = np.load('HR_praat_envelopes.npz', allow_pickle=True)
        idx = {uid: i for i, uid in enumerate(data['utterance_ids'])}
        f0 = data['f0_values'][idx['some_utterance_id']]
    """
    if not uid_envelope_pairs:
        return

    uids = [uid for uid, _ in uid_envelope_pairs]
    array_keys = [
        "f0_times", "f0_values",
        "intensity_times", "intensity_values",
        "word_starts", "word_ends",
        "f0_word_mean", "f0_word_median",
        "intensity_word_mean", "intensity_word_median",
        "f1_word_median", "f2_word_median", "f3_word_median",
    ]

    arrays: dict = {"utterance_ids": np.array(uids, dtype=object)}
    for key in array_keys:
        ragged = []
        for _, env in uid_envelope_pairs:
            if env is None:
                ragged.append(np.array([], dtype=np.float32))
            else:
                raw = env.get(key, [])
                arr = np.array(
                    [v if v is not None else np.nan for v in raw],
                    dtype=np.float32,
                )
                ragged.append(arr)
        arrays[key] = np.array(ragged, dtype=object)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


# Scalar columns written to TSV (envelopes go to separate npz)
_TSV_FEATURE_COLS = [
    "f0_raw", "intensity_raw",
    "f1_median", "f2_median", "f3_median",
    "hnr_median",
    "jitter_local", "jitter_rap",
    "shimmer_local", "shimmer_local_db", "shimmer_apq11",
    "hnr_utt",
]


def main():
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 19)
    except OSError:
        pass
    args = parse_args()
    cfg = load_config(args.config)
    langs = args.langs or cfg["languages"]
    idir = get_intermediate_dir(cfg)
    save_envelopes = cfg.get("praat", {}).get("save_envelopes", False)

    for lang in langs:
        out_tsv = idir / f"{lang}_praat.tsv"
        if out_tsv.exists():
            print(f"[{lang}] {out_tsv} already exists — skipping. Delete to rerun.")
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
              f"({args.workers} workers, save_envelopes={save_envelopes}) ...")

        if args.workers <= 1:
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
        envelope_pairs = []

        for feats in feats_list:
            uid = feats["utterance_id"]
            rec = meta_by_uid[uid]
            row = {
                "utterance_id":   uid,
                "speaker_id":     rec["speaker_id"],
                "session_id":     rec.get("session_id"),
                "language":       lang,
                "sentiment_score":rec["sentiment_score"],
                "sentiment_label":rec["sentiment_label"],
                "n_words":        rec["n_words"],
            }
            for col in _TSV_FEATURE_COLS:
                row[col] = feats.get(col)
            rows.append(row)

            if save_envelopes:
                envelope_pairs.append((uid, feats.get("_envelopes")))

        df = pd.DataFrame(rows)
        n = len(df)
        print(f"  f0_raw valid:        {df['f0_raw'].notna().sum():,}/{n:,}")
        print(f"  intensity_raw valid: {df['intensity_raw'].notna().sum():,}/{n:,}")
        print(f"  f1_median valid:     {df['f1_median'].notna().sum():,}/{n:,}")
        print(f"  jitter_local valid:  {df['jitter_local'].notna().sum():,}/{n:,}")
        print(f"  shimmer_local valid: {df['shimmer_local'].notna().sum():,}/{n:,}")
        print(f"  hnr_utt valid:       {df['hnr_utt'].notna().sum():,}/{n:,}")

        write_tsv(df, out_tsv)
        print(f"  TSV → {out_tsv}")

        if save_envelopes and envelope_pairs:
            npz_path = idir / f"{lang}_praat_envelopes.npz"
            print(f"  Saving envelopes ({len(envelope_pairs):,} utterances) ...")
            _save_envelopes_npz(npz_path, envelope_pairs)
            print(f"  NPZ → {npz_path}")


if __name__ == "__main__":
    main()
