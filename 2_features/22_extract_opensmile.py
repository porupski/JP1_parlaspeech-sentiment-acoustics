#!/usr/bin/env python3
# ============================================================
# Script:  22_extract_opensmile.py
# Release: 1.0
# Version: v1.10
# Purpose: Extract OpenSMILE GeMAPS features per utterance (appendix track).
#          Disabled by default; enable via config enable_opensmile = true.
#
# Input:   {intermediate_dir}/{lang}_filtered.jsonl
# Output:  {intermediate_dir}/{lang}_opensmile.tsv      (GeMAPSv01b functionals)
#          {intermediate_dir}/{lang}_opensmile_lld.npz  (frame-level LLD, if save_opensmile_lld=true)
#
# LLD NPZ keys: utterance_ids, times, f0_lld, loudness_lld, f1_lld, f2_lld, f3_lld, hnr_lld
# All frame arrays are ragged float32 (one 1-D array per utterance, ~10ms frame step).
# F0 is in log semitones from 27.5Hz (GeMAPSv01b convention), voiced frames only.
# Load example:
#   data = np.load('HR_opensmile_lld.npz', allow_pickle=True)
#   idx = {uid: i for i, uid in enumerate(data['utterance_ids'])}
#   f0 = data['f0_lld'][idx['some_id']]   # shape (n_voiced_frames,)
#
# v1.10: Added LLD NPZ output.
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
_G_SAVE_LLD: bool = False
_SMILE = None       # GeMAPSv01b Functionals — lazy-init per worker
_SMILE_LLD = None   # GeMAPSv01b LowLevelDescriptors — lazy-init per worker

# GeMAPSv01b LLD column names for the features we save in the NPZ
_LLD_COL_MAP = {
    "f0_lld":       "F0semitoneFrom27.5Hz_sma3nz",
    "loudness_lld": "Loudness_sma3",
    "f1_lld":       "F1frequency_sma3nz",
    "f2_lld":       "F2frequency_sma3nz",
    "f3_lld":       "F3frequency_sma3nz",
    "hnr_lld":      "HNRdBACF_sma3nz",
}


def _get_smile():
    global _SMILE
    if _SMILE is None:
        import opensmile
        _SMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _SMILE


def _get_smile_lld():
    global _SMILE_LLD
    if _SMILE_LLD is None:
        import opensmile
        _SMILE_LLD = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
    return _SMILE_LLD


def _init_worker(index: dict, audio_root_str: str, save_lld: bool) -> None:
    global _G_INDEX, _G_ROOT, _G_SAVE_LLD
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 19)
    except OSError:
        pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _G_INDEX = index
    _G_ROOT = Path(audio_root_str)
    _G_SAVE_LLD = save_lld


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

    result: dict = {"utterance_id": uid}
    try:
        smile = _get_smile()
        feats = smile.process_file(str(audio_path), start=start, end=end)
        row = feats.iloc[0].to_dict()
        result.update({f"osmile_{k}": v for k, v in row.items()})
    except Exception:
        pass

    if _G_SAVE_LLD:
        try:
            smile_lld = _get_smile_lld()
            lld_df = smile_lld.process_file(str(audio_path), start=start, end=end)
            # Extract time axis from MultiIndex (start_time per frame)
            times = np.array(
                [idx[1] for idx in lld_df.index], dtype=np.float32
            )
            lld_data: dict[str, np.ndarray] = {"_lld_times": times}
            for key, col in _LLD_COL_MAP.items():
                if col in lld_df.columns:
                    arr = lld_df[col].values.astype(np.float32)
                    arr[np.isinf(arr)] = np.nan
                else:
                    arr = np.full(len(times), np.nan, dtype=np.float32)
                lld_data[key] = arr
            result["_lld"] = lld_data
        except Exception:
            result["_lld"] = None

    return result


def _save_lld_npz(path: Path, uid_lld_pairs: list) -> None:
    """Save frame-level LLD arrays to a single compressed NPZ (ragged object arrays)."""
    if not uid_lld_pairs:
        return
    uids = [uid for uid, _ in uid_lld_pairs]
    keys = ["_lld_times"] + list(_LLD_COL_MAP.keys())
    arrays: dict = {"utterance_ids": np.array(uids, dtype=object)}
    npz_key_map = {"_lld_times": "times"}
    for key in keys:
        npz_key = npz_key_map.get(key, key)
        ragged = []
        for _, lld in uid_lld_pairs:
            if lld is None:
                ragged.append(np.array([], dtype=np.float32))
            else:
                ragged.append(lld.get(key, np.array([], dtype=np.float32)))
        arrays[npz_key] = np.array(ragged, dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


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

    if not cfg["analysis"]["enable_opensmile"]:
        print("OpenSMILE disabled in config (enable_opensmile = false). Exiting.")
        return

    save_lld = cfg["analysis"].get("save_opensmile_lld", False)
    langs = args.langs or cfg["languages"]
    idir = get_intermediate_dir(cfg)

    for lang in langs:
        out_tsv = idir / f"{lang}_opensmile.tsv"
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
        print(f"\n[{lang}] Extracting OpenSMILE GeMAPS from {len(records):,} utterances "
              f"({args.workers} workers, save_lld={save_lld}) ...")

        if args.workers <= 1:
            global _G_INDEX, _G_ROOT, _G_SAVE_LLD
            _G_INDEX = index
            _G_ROOT = audio_root
            _G_SAVE_LLD = save_lld
            feats_list = [_process_record_worker(r) for r in tqdm(records, desc=lang)]
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(index, str(audio_root), save_lld),
            ) as pool:
                feats_list = list(tqdm(
                    pool.map(_process_record_worker, records, chunksize=50),
                    total=len(records),
                    desc=lang,
                ))

        # Separate scalar TSV rows from LLD data
        lld_pairs = []
        tsv_rows = []
        for r in feats_list:
            lld = r.pop("_lld", None)
            lld_pairs.append((r["utterance_id"], lld))
            tsv_rows.append({k: v for k, v in r.items() if not k.startswith("_")})

        df = pd.DataFrame(tsv_rows)
        write_tsv(df, out_tsv)
        print(f"  TSV → {out_tsv} ({len(df.columns)} cols)")

        if save_lld and any(lld is not None for _, lld in lld_pairs):
            npz_path = idir / f"{lang}_opensmile_lld.npz"
            print(f"  Saving LLD NPZ ({len(lld_pairs):,} utterances) ...")
            _save_lld_npz(npz_path, lld_pairs)
            print(f"  LLD NPZ → {npz_path}")


if __name__ == "__main__":
    main()
