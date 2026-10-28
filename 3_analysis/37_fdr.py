#!/usr/bin/env python3
# ============================================================
# Script:  37_fdr.py
# Release: 1.0
# Version: v1.00
# Purpose: Family-wise Benjamini-Hochberg FDR correction pooling ALL
#          p-values across H1, H2, H3 into one family. The per-script
#          BH already written by 30/31/32 is within-hypothesis; this
#          script answers the stricter reviewer question: how many
#          tests survive once the whole paper's test family is pooled?
#
# Family (84 tests):
#   H1: 12 lang×feat cells × 2 tracks (speaker_avg, utterance_level)   = 24
#   H2: 12 lang×feat cells × 1 (per-speaker Kendall full range)        = 12
#   H3: 12 lang×feat cells × 2 sides × 2 tests (Kendall + linear)      = 48
#
# Input:   {results_dir}/h1_results.json
#          {results_dir}/h2_results.json
#          {results_dir}/h3_results.json  (split method only)
# Output:  {results_dir}/family_fdr.json
#            Flat list of {hypothesis, cell, test, p_raw, p_bh_family}
#            + survival summary at alpha levels 0.05 and 0.01.
#          Prints a human-readable breakdown.
# ============================================================

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_results_dir
from utils.stats import bh_correct


def _collect_h1(results: dict) -> list[dict]:
    rows = []
    for cell, v in results.items():
        if not isinstance(v, dict):
            continue
        for track in ("speaker_avg", "utterance_level"):
            sub = v.get(track, {})
            p = sub.get("p", None)
            if p is None or (isinstance(p, float) and np.isnan(p)):
                continue
            rows.append({"hypothesis": "H1", "cell": cell, "test": track,
                         "p_raw": float(p)})
    return rows


def _collect_h2(results: dict) -> list[dict]:
    rows = []
    for cell, v in results.items():
        if not isinstance(v, dict):
            continue
        p = v.get("p", None)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        rows.append({"hypothesis": "H2", "cell": cell, "test": "kendall_full",
                     "p_raw": float(p)})
    return rows


def _collect_h3(results: dict) -> list[dict]:
    rows = []
    for cell, v in results.items():
        if cell.startswith("_") or not isinstance(v, dict):
            continue
        for side in ("neg", "pos"):
            sub = v.get(side, {})
            for test_name, p_key in [("kendall", "kendall_p"),
                                     ("linear",  "linear_p")]:
                p = sub.get(p_key, None)
                if p is None or (isinstance(p, float) and np.isnan(p)):
                    continue
                rows.append({"hypothesis": "H3", "cell": cell,
                             "test": f"{side}_{test_name}",
                             "p_raw": float(p)})
    return rows


def _survival_counts(rows: list[dict], alpha: float) -> dict:
    """Count tests surviving at given alpha, raw vs family-corrected."""
    by_h = {"H1": [0, 0, 0], "H2": [0, 0, 0], "H3": [0, 0, 0]}  # [raw, bh, total]
    for r in rows:
        h = r["hypothesis"]
        by_h[h][2] += 1
        if r["p_raw"] < alpha:
            by_h[h][0] += 1
        if r["p_bh_family"] < alpha:
            by_h[h][1] += 1
    return by_h


def main():
    cfg = load_config()
    rdir = get_results_dir(cfg)
    alpha = cfg["analysis"]["bh_alpha"]

    paths = {
        "H1": rdir / "h1_results.json",
        "H2": rdir / "h2_results.json",
        "H3": rdir / "h3_results.json",
    }
    missing = [h for h, p in paths.items() if not p.exists()]
    if missing:
        print(f"Missing result files: {missing}. Run 30/31/32 first.")
        sys.exit(1)

    with open(paths["H1"]) as f:
        h1 = json.load(f)
    with open(paths["H2"]) as f:
        h2 = json.load(f)
    with open(paths["H3"]) as f:
        h3 = json.load(f)

    # H3 family-wise correction is only meaningful for the fixed-inflection
    # split method, which is what the paper reports. Quadratic uses a
    # different test and should be corrected on its own.
    h3_method = h3.get("_method", "split")
    if h3_method != "split":
        print(f"NOTE: H3 method is '{h3_method}', not 'split'. "
              f"Family correction will skip H3.")
        rows = _collect_h1(h1) + _collect_h2(h2)
    else:
        rows = _collect_h1(h1) + _collect_h2(h2) + _collect_h3(h3)

    pvals = [r["p_raw"] for r in rows]
    corrected = bh_correct(pvals, alpha=alpha)
    for r, p_bh in zip(rows, corrected):
        r["p_bh_family"] = float(p_bh)

    # Per-hypothesis BH (sanity check against inline corrections in 30/31/32)
    for h_name in ("H1", "H2", "H3"):
        subset = [r for r in rows if r["hypothesis"] == h_name]
        if not subset:
            continue
        sub_corr = bh_correct([r["p_raw"] for r in subset], alpha=alpha)
        for r, p_bh in zip(subset, sub_corr):
            r["p_bh_within"] = float(p_bh)

    survival_05 = _survival_counts(rows, 0.05)
    survival_01 = _survival_counts(rows, 0.01)

    print(f"\n{'='*60}")
    print(f"Family-wise FDR summary — {len(rows)} tests pooled")
    print(f"{'='*60}")
    print(f"{'':6s}  {'raw<.05':>8s}  {'BH<.05':>8s}  "
          f"{'raw<.01':>8s}  {'BH<.01':>8s}  {'total':>6s}")
    for h in ("H1", "H2", "H3"):
        s5 = survival_05[h]
        s1 = survival_01[h]
        if s5[2] == 0:
            continue
        print(f"{h:6s}  {s5[0]:>8d}  {s5[1]:>8d}  "
              f"{s1[0]:>8d}  {s1[1]:>8d}  {s5[2]:>6d}")

    tot5 = [sum(c) for c in zip(*survival_05.values())]
    tot1 = [sum(c) for c in zip(*survival_01.values())]
    print(f"{'ALL':6s}  {tot5[0]:>8d}  {tot5[1]:>8d}  "
          f"{tot1[0]:>8d}  {tot1[1]:>8d}  {tot5[2]:>6d}")

    out = {
        "_alpha":   alpha,
        "_n_tests": len(rows),
        "_h3_method": h3_method,
        "_survival_05": {h: {"raw": v[0], "bh_family": v[1], "total": v[2]}
                          for h, v in survival_05.items()},
        "_survival_01": {h: {"raw": v[0], "bh_family": v[1], "total": v[2]}
                          for h, v in survival_01.items()},
        "tests": rows,
    }
    out_path = rdir / "family_fdr.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: None if (isinstance(x, float)
                                              and np.isnan(x)) else x)
    print(f"\nWritten → {out_path}")


if __name__ == "__main__":
    main()
