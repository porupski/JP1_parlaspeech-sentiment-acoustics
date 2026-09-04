#!/usr/bin/env python3
# ============================================================
# Script:  35_vad.py
# Release: 1.0
# Version: v1.00
# Purpose: Compute utterance-level VAD (Valence-Arousal-Dominance) scores
#          from text using the NRC VAD Lexicon (multilingual).
#          Correlate with ParlaSent sentiment scores per language.
#
#          This is the empirical bridge requested by the reviewer:
#          "The correlation between sentiment and VAD would provide
#           guidance on what to expect from acoustic features."
#
# Config:  vad.enable_text_vad — enable/disable this script
#          vad.nrc_vad_path    — path to NRC-VAD-Lexicon.txt
#          vad.nrc_lang_codes  — mapping from our lang codes to NRC column names
#
# NRC VAD Lexicon download:
#   https://saifmohammad.com/WebPages/nrclex.html
#   File: NRC-VAD-Lexicon.zip → NRC-VAD-Lexicon.txt
#   Format: Word  Valence  Arousal  Dominance  [lang columns...]
#
# Input:   {intermediate_dir}/{lang}_filtered.jsonl (for text)
#          + NRC VAD Lexicon
# Output:  {results_dir}/vad_correlations.json
#          {intermediate_dir}/{lang}_vad.tsv (utterance-level VAD scores)
# ============================================================

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir, get_results_dir
from utils.data_utils import load_jsonl, write_tsv


def load_nrc_vad(path: str | Path, lang_col: str) -> dict[str, dict]:
    """
    Load NRC VAD lexicon for a given language column.
    Returns dict: {lowercase_word: {'valence': float, 'arousal': float, 'dominance': float}}

    The multilingual NRC VAD file has a header row with language names as columns.
    English valence/arousal/dominance are the first three value columns.
    Other languages have their words in dedicated columns.

    If lang_col not found, falls back to English word list.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.columns = df.columns.str.strip()

    # Expected English columns
    val_col = "Valence"
    aro_col = "Arousal"
    dom_col = "Dominance"
    word_col = "Word"

    # Some versions of NRC use different column names — adjust as needed
    if word_col not in df.columns:
        word_col = df.columns[0]

    lexicon = {}
    for _, row in df.iterrows():
        word = str(row.get(word_col, "")).lower().strip()
        if not word:
            continue
        try:
            v = float(row.get(val_col, np.nan))
            a = float(row.get(aro_col, np.nan))
            d = float(row.get(dom_col, np.nan))
        except (ValueError, TypeError):
            continue
        if not np.isnan(v):
            lexicon[word] = {"valence": v, "arousal": a, "dominance": d}

    return lexicon


def utterance_vad(text: str, lexicon: dict) -> dict:
    """
    Mean VAD across content words found in lexicon.
    Returns {valence, arousal, dominance, n_covered}
    """
    words = [w.lower().strip(".,!?;:\"'()") for w in text.split()]
    covered = [lexicon[w] for w in words if w in lexicon]
    if not covered:
        return {"valence": np.nan, "arousal": np.nan,
                "dominance": np.nan, "vad_n_covered": 0}
    return {
        "valence": float(np.mean([c["valence"] for c in covered])),
        "arousal": float(np.mean([c["arousal"] for c in covered])),
        "dominance": float(np.mean([c["dominance"] for c in covered])),
        "vad_n_covered": len(covered),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if not cfg["analysis"]["enable_text_vad"]:
        print("Text VAD disabled (enable_text_vad = false). Exiting.")
        return

    vad_cfg = cfg["vad"]
    nrc_path = Path(vad_cfg["nrc_vad_path"])
    if not nrc_path.exists():
        print(f"NRC VAD Lexicon not found at: {nrc_path}")
        print("Download from https://saifmohammad.com/WebPages/nrclex.html")
        return

    langs = args.langs or cfg["languages"]
    idir = get_intermediate_dir(cfg)
    rdir = get_results_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    corr_method = vad_cfg.get("vad_corr_method", "spearman")

    lang_codes = vad_cfg.get("nrc_lang_codes", {})
    correlations = {}

    for lang in langs:
        in_path = idir / f"{lang}_filtered.jsonl"
        if not in_path.exists():
            print(f"[{lang}] {in_path} not found. Skipping.")
            continue

        lang_col = lang_codes.get(lang, "English")
        print(f"\n[{lang}] Loading NRC VAD for '{lang_col}' ...")
        lexicon = load_nrc_vad(nrc_path, lang_col)
        print(f"  Lexicon size: {len(lexicon):,} words")

        records = load_jsonl(in_path)
        print(f"  Computing VAD for {len(records):,} utterances ...")

        rows = []
        for rec in records:
            text = rec.get("text", "")
            vad = utterance_vad(text, lexicon)
            rows.append({
                "utterance_id": rec["utterance_id"],
                "sentiment_score": rec["sentiment_score"],
                **vad,
            })

        df = pd.DataFrame(rows)
        valid = df.dropna(subset=["valence", "sentiment_score"])
        coverage = len(valid) / len(df)
        print(f"  VAD coverage: {len(valid):,}/{len(df):,} ({100*coverage:.1f}%)")

        # Spearman correlations: sentiment_score vs {valence, arousal, dominance}
        lang_corr = {}
        for dim in ["valence", "arousal", "dominance"]:
            sub = df.dropna(subset=["sentiment_score", dim])
            if len(sub) < 10:
                lang_corr[dim] = {"r": np.nan, "p": np.nan, "n": len(sub)}
                continue
            if corr_method == "spearman":
                r, p = stats.spearmanr(sub["sentiment_score"], sub[dim])
            else:
                r, p = stats.pearsonr(sub["sentiment_score"], sub[dim])
            lang_corr[dim] = {"r": float(r), "p": float(p), "n": len(sub)}
            print(f"  sentiment ~ {dim}: r={r:.3f}, p={p:.4f}, n={len(sub):,}")

        correlations[lang] = lang_corr

        # Save utterance-level VAD scores
        vad_out = idir / f"{lang}_vad.tsv"
        write_tsv(df, vad_out)

    # Save correlation table
    out = rdir / "vad_correlations.json"
    with open(out, "w") as f:
        json.dump(correlations, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\nWritten → {out}")


if __name__ == "__main__":
    main()
