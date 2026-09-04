#!/usr/bin/env python3
# ============================================================
# Script:  40_tables.py
# Release: 1.0
# Version: v1.00
# Purpose: Generate LaTeX tables from analysis results JSONs.
#
# Output:  {results_dir}/tables/tab_h1.tex
#          {results_dir}/tables/tab_h2.tex
#          {results_dir}/tables/tab_h3.tex
#          {results_dir}/tables/tab_dataset_stats.tex
# ============================================================

import sys
import json
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_results_dir, get_intermediate_dir
from utils.latex_utils import (format_table_h1, format_table_h2,
                                 format_table_h3, write_table)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    return p.parse_args()


def make_dataset_stats_table(cfg: dict, idir: Path) -> str:
    """Table 1: dataset statistics before and after filtering."""
    langs = cfg["languages"]
    rows = []
    for lang in langs:
        filt_path = idir / f"{lang}_filtered.jsonl"
        raw_path = None  # raw counts would need to be stored during filter step
        n_filtered = sum(1 for _ in open(filt_path)) if filt_path.exists() else "?"
        rows.append({"lang": lang, "n_filtered": n_filtered})

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"\textbf{Language} & \textbf{Utterances (filtered)} & \textbf{Speakers} \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(f"{row['lang']} & {row['n_filtered']} & --- \\\\")
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{Dataset statistics after filtering (v4.0).}",
              r"\label{tab:dataset_stats}", r"\end{table}"]
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    rdir = get_results_dir(cfg)
    idir = get_intermediate_dir(cfg)
    tdir = rdir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    langs = cfg["languages"]
    feats_main = cfg["analysis"]["features_main"]
    feats_all = feats_main + cfg["analysis"]["features_appendix"]

    def load_json(name):
        path = rdir / name
        if not path.exists():
            print(f"  Missing: {path}")
            return {}
        with open(path) as f:
            return json.load(f)

    # Reshape JSON results from "LANG_feat" keys → (lang, feat) tuple keys
    def reshape(d):
        out = {}
        for k, v in d.items():
            parts = k.split("_", 1)
            if len(parts) == 2:
                out[(parts[0], parts[1])] = v
        return out

    print("Generating H1 table ...")
    h1 = reshape(load_json("h1_results.json"))
    write_table(format_table_h1(h1, feats_all, langs), tdir / "tab_h1.tex")

    print("Generating H2 table ...")
    h2 = reshape(load_json("h2_results.json"))
    write_table(format_table_h2(h2, feats_all, langs), tdir / "tab_h2.tex")

    print("Generating H3 table ...")
    h3 = reshape(load_json("h3_results.json"))
    write_table(format_table_h3(h3, feats_all, langs), tdir / "tab_h3.tex")

    print("Generating dataset stats table ...")
    write_table(make_dataset_stats_table(cfg, idir), tdir / "tab_dataset_stats.tex")

    print(f"\nAll tables → {tdir}")


if __name__ == "__main__":
    main()
