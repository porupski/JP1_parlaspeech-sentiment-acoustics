#!/usr/bin/env python3
# ============================================================
# Script:  22_extract_opensmile.py
# Release: 1.0
# Version: v1.01
# Purpose: Extract OpenSMILE GeMAPS features per utterance (appendix track).
#          Disabled by default; enable via config enable_opensmile = true.
#
# Input:   {intermediate_dir}/{lang}_filtered.jsonl
# Output:  {intermediate_dir}/{lang}_opensmile.tsv
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


# Worker-process globals set by _init_worker
_G_INDEX: dict = {}
_G_ROOT: "Path | None" = None
_SMILE = None  # lazy-init per worker process


def _get_smile():
    global _SMILE
    if _SMILE is None:
        import opensmile
        _SMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _SMILE


def _init_worker(index: dict, audio_root_str: str) -> None:
    global _G_INDEX, _G_ROOT
    os.nice(19)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _G_INDEX = index
    _G_ROOT = Path(audio_root_str)


def _process_record_worker(rec: dict) -> dict:
    uid = rec["utterance_id"]
    audio_rel = rec.get("audio")
    if not audio_rel:
        return {"utterance_id": uid}
    audio_path = resolve_audio_path(audio_rel, _G_ROOT, _G_INDEX)
    if audio_path is None:
        return {"utterance_id": uid}
    words_align = rec.get("words_align", [])
    if not words_align:
        return {"utterance_id": uid}
    start = words_align[0]["start"]
    end = words_align[-1]["end"]
    try:
        smile = _get_smile()
        feats = smile.process_file(str(audio_path), start=start, end=end)
        row = feats.iloc[0].to_dict()
        return {"utterance_id": uid, **{f"osmile_{k}": v for k, v in row.items()}}
    except Exception:
        return {"utterance_id": uid}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel worker processes for audio extraction")
    return p.parse_args()


def main():
    os.nice(19)
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
        print(f"\n[{lang}] Building audio index from {audio_root} ...")
        index = build_audio_index(audio_root)
        print(f"[{lang}] Index: {len(index):,} audio files")
        print(f"\n[{lang}] Extracting OpenSMILE GeMAPS from {len(records):,} utterances "
              f"({args.workers} workers) ...")

        if args.workers <= 1:
            global _G_INDEX, _G_ROOT
            _G_INDEX = index
            _G_ROOT = audio_root
            feats_list = [_process_record_worker(r) for r in tqdm(records, desc=lang)]
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(index, str(audio_root)),
            ) as pool:
                feats_list = list(tqdm(
                    pool.map(_process_record_worker, records, chunksize=50),
                    total=len(records),
                    desc=lang,
                ))

        df = pd.DataFrame(feats_list)
        out = idir / f"{lang}_opensmile.tsv"
        write_tsv(df, out)
        print(f"  Written → {out} ({len(df.columns)} cols)")


if __name__ == "__main__":
    main()
