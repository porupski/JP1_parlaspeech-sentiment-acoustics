#!/usr/bin/env python3
# ============================================================
# Script:  41_numbers.py
# Release: 1.0
# Version: v1.00
# Purpose: Emit results/numbers.json — every in-text number, keyed.
#          The LaTeX paper reads from this file; no hand-typed stats.
#
# Output:  {results_dir}/numbers.json
# ============================================================

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_results_dir
from utils.latex_utils import build_numbers_json, write_numbers_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    rdir = get_results_dir(cfg)
    langs = cfg["languages"]
    feats = cfg["analysis"]["features_main"] + cfg["analysis"]["features_appendix"]

    def load_json(name):
        path = rdir / name
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def reshape(d):
        out = {}
        for k, v in d.items():
            parts = k.split("_", 1)
            if len(parts) == 2:
                out[(parts[0], parts[1])] = v
        return out

    h1 = reshape(load_json("h1_results.json"))
    h2 = reshape(load_json("h2_results.json"))
    h3 = reshape(load_json("h3_results.json"))

    numbers = build_numbers_json(h1, h2, h3, feats, langs)

    # Global trend split point
    gt = load_json("global_trend.json")
    numbers["split_point"] = gt.get("split_point")
    numbers["split_bin"] = gt.get("split_bin")

    # VAD correlations (sentiment ~ valence, most important one for paper)
    vad = load_json("vad_correlations.json")
    for lang in langs:
        for dim in ["valence", "arousal", "dominance"]:
            r = vad.get(lang, {}).get(dim, {})
            numbers[f"vad_{dim}_{lang}_r"] = r.get("r")
            numbers[f"vad_{dim}_{lang}_p"] = r.get("p")
            numbers[f"vad_{dim}_{lang}_n"] = r.get("n")

    out = rdir / "numbers.json"
    write_numbers_json(numbers, out)
    print(f"Written → {out}")
    print(f"  {len(numbers)} keys")

    # Print the headline numbers
    print(f"\nHeadline numbers:")
    print(f"  H1 sig (speaker avg): {numbers.get('h1_sig_speaker_avg')}/{numbers.get('h1_total')}")
    print(f"  H2 sig: {numbers.get('h2_sig')}/{numbers.get('h2_total')}")
    print(f"  H3 strong: {numbers.get('h3_strong')} | partial: {numbers.get('h3_partial')} / {numbers.get('h3_total')}")
    print(f"  Split point: {numbers.get('split_point')}")


if __name__ == "__main__":
    main()
