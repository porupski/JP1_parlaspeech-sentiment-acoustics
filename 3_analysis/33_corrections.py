#!/usr/bin/env python3
# ============================================================
# Script:  33_corrections.py
# Release: 1.0
# Version: v1.00
# Purpose: Post-hoc: verify BH correction was applied (done inline in
#          30/31/32), print a consolidated correction summary, and
#          report how many results survive BH at alpha=0.05.
#
#          Also a good sanity check to run after all analysis scripts.
#
# Input:   {results_dir}/h1_results.json
#          {results_dir}/h2_results.json
#          {results_dir}/h3_results.json
# Output:  Prints summary; no new files (corrections already in JSONs).
# ============================================================

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_results_dir


def count_sig(results: dict, p_key: str = "p", alpha: float = 0.05) -> tuple[int, int]:
    """Count how many results are significant (raw) vs NaN total."""
    total, sig = 0, 0
    for v in results.values():
        if isinstance(v, dict):
            p = v.get(p_key, np.nan)
            if p is not None and not np.isnan(p):
                total += 1
                if p < alpha:
                    sig += 1
    return sig, total


def main():
    cfg = load_config()
    rdir = get_results_dir(cfg)
    alpha = cfg["analysis"]["bh_alpha"]

    for name, path_name in [("H1", "h1_results.json"),
                              ("H2", "h2_results.json"),
                              ("H3", "h3_results.json")]:
        path = rdir / path_name
        if not path.exists():
            print(f"{name}: {path} not found")
            continue
        with open(path) as f:
            results = json.load(f)

        print(f"\n{'='*50}")
        print(f"{name} correction summary (alpha={alpha})")
        print(f"{'='*50}")

        if name == "H1":
            sa_sig, sa_tot = 0, 0
            sa_bh_sig, ul_sig, ul_tot, ul_bh_sig = 0, 0, 0, 0
            for k, v in results.items():
                for track, label in [("speaker_avg", "SA"), ("utterance_level", "UL")]:
                    sub = v.get(track, {})
                    p = sub.get("p", np.nan)
                    p_bh = sub.get("p_bh", np.nan)
                    if p is None or np.isnan(p):
                        continue
                    if track == "speaker_avg":
                        sa_tot += 1
                        if p < alpha: sa_sig += 1
                        if not np.isnan(p_bh) and p_bh < alpha: sa_bh_sig += 1
                    else:
                        ul_tot += 1
                        if p < alpha: ul_sig += 1
                        if not np.isnan(p_bh) and p_bh < alpha: ul_bh_sig += 1
            print(f"  Speaker avg:      {sa_sig}/{sa_tot} raw sig | "
                  f"{sa_bh_sig}/{sa_tot} after BH")
            print(f"  Utterance level:  {ul_sig}/{ul_tot} raw sig | "
                  f"{ul_bh_sig}/{ul_tot} after BH")

        elif name == "H2":
            raw_sig, tot = 0, 0
            bh_sig = 0
            for k, v in results.items():
                p = v.get("p", np.nan)
                p_bh = v.get("p_bh", np.nan)
                if p is None or np.isnan(p): continue
                tot += 1
                if p < alpha: raw_sig += 1
                if not np.isnan(p_bh) and p_bh < alpha: bh_sig += 1
            print(f"  Kendall tau:  {raw_sig}/{tot} raw sig | {bh_sig}/{tot} after BH")

        elif name == "H3":
            checks = {"strong": 0, "partial": 0, "none": 0}
            for k, v in results.items():
                chk = v.get("check", "none")
                checks[chk] = checks.get(chk, 0) + 1
            n = sum(checks.values())
            print(f"  Strong arousal:   {checks['strong']}/{n}")
            print(f"  Partial arousal:  {checks['partial']}/{n}")
            print(f"  No arousal:       {checks['none']}/{n}")
            print(f"  Supported (strong+partial): {checks['strong']+checks['partial']}/{n}")


if __name__ == "__main__":
    main()
