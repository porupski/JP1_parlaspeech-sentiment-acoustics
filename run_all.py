#!/usr/bin/env python3
# ============================================================
# Script:  run_all.py
# Release: 1.0
# Version: v1.00
# Purpose: End-to-end pipeline runner. Executes all stages in order.
#          Each stage can be skipped with --skip or run alone with --only.
#          Reads config.json for all parameters.
#
# Usage:
#   python run_all.py                          # run all stages
#   python run_all.py --langs HR CZ            # specific languages
#   python run_all.py --skip filter join       # skip stages by name
#   python run_all.py --only h2 h3 tables      # run only these stages
#   python run_all.py --dry-run                # filter step only, no writes
# ============================================================

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


STAGES = {
    "filter":     ("1_data/10_filter.py",           "Filter JSONL by word count + speaker coverage"),
    "speechrate": ("2_features/21_extract_speechrate.py", "Extract speech rate features"),
    "praat":      ("2_features/20_extract_praat.py", "Extract Praat F0 + intensity"),
    "opensmile":  ("2_features/22_extract_opensmile.py",  "Extract OpenSMILE GeMAPS (appendix)"),
    "normalize":  ("2_features/23_normalize.py",     "Add per-session/speaker normalized columns"),
    "join":       ("1_data/11_join.py",              "Join all feature TSVs per language"),
    "h1":         ("3_analysis/30_h1_extremes.py",   "H1: Wilcoxon extremes"),
    "h2":         ("3_analysis/31_h2_monotonic.py",  "H2: Kendall tau"),
    "h3":         ("3_analysis/32_h3_split.py",      "H3: Split analysis"),
    "corrections":("3_analysis/33_corrections.py",   "Print BH correction summary"),
    "gamm":       ("3_analysis/34_gamm.py",          "GAMM (optional, booleaned off)"),
    "vad":        ("3_analysis/35_vad.py",            "VAD correlation on text"),
    "tables":     ("4_outputs/40_tables.py",          "Generate LaTeX tables"),
    "numbers":    ("4_outputs/41_numbers.py",          "Emit numbers.json"),
    "plots":      ("4_outputs/42_plots.py",            "Generate all figures"),
}

DEFAULT_ORDER = [
    "filter", "speechrate", "praat", "opensmile",
    "normalize", "join",
    "h1", "h2", "h3", "corrections", "vad",
    "tables", "numbers", "plots",
]


def parse_args():
    p = argparse.ArgumentParser(description="Run full pipeline")
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    p.add_argument("--skip", nargs="+", default=[], choices=list(STAGES.keys()),
                   help="Stages to skip")
    p.add_argument("--only", nargs="+", default=[], choices=list(STAGES.keys()),
                   help="Run only these stages (overrides --skip)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel worker processes for Praat/OpenSMILE extraction")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_stage(name: str, script: str, extra_args: list[str]) -> bool:
    cmd = [sys.executable, script, "--config", "config.json"] + extra_args
    print(f"\n{'='*60}")
    print(f"STAGE: {name.upper()}")
    print(f"CMD:   {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


WORKER_STAGES = {"praat", "opensmile"}


def main():
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 19)
    except OSError:
        pass
    args = parse_args()
    lang_args = (["--langs"] + args.langs) if args.langs else []

    if args.only:
        order = [s for s in DEFAULT_ORDER if s in args.only]
    else:
        order = [s for s in DEFAULT_ORDER if s not in args.skip]

    print(f"\nParlaSpeech Sentiment–Acoustics Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stages:  {' → '.join(order)}")
    print(f"Workers: {args.workers} (for praat/opensmile stages)")

    failed = []
    for name in order:
        script, desc = STAGES[name]
        extra = lang_args.copy()
        if args.dry_run and name == "filter":
            extra.append("--dry-run")
        if name in WORKER_STAGES:
            extra += ["--workers", str(args.workers)]
        ok = run_stage(name, script, extra)
        if not ok:
            print(f"\n[ERROR] Stage '{name}' failed. Stopping.")
            failed.append(name)
            break

    print(f"\n{'='*60}")
    if failed:
        print(f"PIPELINE FAILED at: {failed}")
    else:
        print(f"PIPELINE COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
