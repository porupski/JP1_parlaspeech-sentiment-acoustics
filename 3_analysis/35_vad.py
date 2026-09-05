#!/usr/bin/env python3
# ============================================================
# Script:  35_vad.py
# Release: 1.0
# Version: v1.10
# Purpose: Compute utterance-level VAD (Valence-Arousal-Dominance) scores
#          from text and correlate with ParlaSent sentiment scores per language.
#
# Method:
#   1. Load NRC VAD v1 per-language lexicon (OneFilePerLanguage/ directory).
#      Format: English Word | Valence | Arousal | Dominance | Translated Word
#      Lookup key = translated word (last column), scores from English entry.
#   2. For each filtered utterance, fetch lemmas from the original v4 JSONL
#      linguistic_annotation field (already computed by ParlaSpeech pipeline).
#      Filter to content words (UPOS in vad.vad_content_upos).
#   3. Compute mean VAD across matched lemmas.
#   4. Spearman correlate utterance-level VAD with ParlaSent score.
#
# Notes on language proxies:
#   RS (Serbian): NRC v1 Serbian file is Cyrillic; ParlaSpeech RS is Latin.
#      Using Bosnian (Latin, mutually intelligible) as proxy. See config RS note.
#
# Input:   {intermediate_dir}/{lang}_filtered.jsonl  (utterance IDs + sentiment)
#          {data_root}/ParlaSpeech-{lang}.v4.0.patched.jsonl  (lemmas)
#          {nrc_vad_dir}/{LangName}-NRC-VAD-Lexicon.txt
# Output:  {intermediate_dir}/{lang}_vad.tsv
#          {results_dir}/vad_correlations.json
#
# v1.10: Rewrote for NRC VAD v1 per-language file format.
#        Uses lemmas from v4 JSONL linguistic_annotation (no CLASSLA/Stanza needed).
#        Added UPOS content-word filtering.
# ============================================================

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_intermediate_dir, get_results_dir, get_jsonl_path
from utils.data_utils import load_jsonl, write_tsv


def load_nrc_vad_v1(vad_dir: Path, lang_name: str) -> dict[str, dict]:
    """
    Load NRC VAD v1 per-language lexicon.

    File: {lang_name}-NRC-VAD-Lexicon.txt
    Columns (tab-separated): English Word | Valence | Arousal | Dominance | Translated Word

    Returns dict: {lowercase_translated_word: {'valence': float, 'arousal': float, 'dominance': float}}
    Translated word is the lookup key; VAD scores are from the English-annotated entry.
    """
    fpath = vad_dir / f"{lang_name}-NRC-VAD-Lexicon.txt"
    if not fpath.exists():
        raise FileNotFoundError(f"NRC VAD lexicon not found: {fpath}")

    lexicon: dict[str, dict] = {}
    with open(fpath, encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            try:
                valence   = float(parts[1])
                arousal   = float(parts[2])
                dominance = float(parts[3])
            except ValueError:
                continue
            translated = parts[4].strip().lower()
            if not translated:
                continue
            # Multi-word translations: index each token individually (last wins on collision)
            for token in translated.split():
                token = token.strip()
                if token:
                    lexicon[token] = {"valence": valence, "arousal": arousal,
                                      "dominance": dominance}
            # Also index the full phrase for exact multi-word matches
            if " " in translated:
                lexicon[translated] = {"valence": valence, "arousal": arousal,
                                       "dominance": dominance}
    return lexicon


def _utterance_vad(lemmas: list[str], lexicon: dict) -> dict:
    """Mean VAD over lemmas found in lexicon. Returns NaN dims if no matches."""
    covered = [lexicon[lm] for lm in lemmas if lm in lexicon]
    if not covered:
        return {"valence": np.nan, "arousal": np.nan,
                "dominance": np.nan, "vad_n_covered": 0}
    return {
        "valence":       float(np.mean([c["valence"]   for c in covered])),
        "arousal":       float(np.mean([c["arousal"]   for c in covered])),
        "dominance":     float(np.mean([c["dominance"] for c in covered])),
        "vad_n_covered": len(covered),
    }


def _load_lemmas_from_v4(jsonl_path: Path,
                           target_ids: set[str],
                           content_upos: set[str]) -> dict[str, list[str]]:
    """
    Stream v4 JSONL and extract lemmas for target utterance IDs.
    Uses linguistic_annotation field (already computed by ParlaSpeech pipeline).
    Filters to content_upos (NOUN, VERB, ADJ, ADV by default).

    Returns dict: {utterance_id: [lemma, ...]}
    """
    lemma_map: dict[str, list[str]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if len(lemma_map) >= len(target_ids):
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get("id")
            if uid not in target_ids:
                continue
            ling = rec.get("linguistic_annotation", [])
            lemmas = [
                tok["lemma"].lower()
                for tok in ling
                if tok.get("upos") in content_upos
                and tok.get("lemma")
                and tok.get("words_idx") is not None  # skip punctuation tokens
            ]
            lemma_map[uid] = lemmas
    return lemma_map


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

    vad_cfg   = cfg["vad"]
    vad_dir   = Path(vad_cfg["nrc_vad_dir"])
    lang_codes = vad_cfg.get("nrc_lang_codes", {})
    content_upos = set(vad_cfg.get("vad_content_upos", ["NOUN", "VERB", "ADJ", "ADV"]))
    corr_method  = vad_cfg.get("vad_corr_method", "spearman")

    if not vad_dir.exists():
        print(f"NRC VAD directory not found: {vad_dir}")
        print("Run: bash 0_env/get_nrc_vad.sh")
        return

    langs  = args.langs or cfg["languages"]
    idir   = get_intermediate_dir(cfg)
    rdir   = get_results_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)

    correlations: dict = {}

    for lang in langs:
        filt_path = idir / f"{lang}_filtered.jsonl"
        if not filt_path.exists():
            print(f"[{lang}] {filt_path} not found. Run 10_filter.py first.")
            continue

        lang_name = lang_codes.get(lang)
        if not lang_name:
            print(f"[{lang}] No nrc_lang_codes entry. Skipping.")
            continue

        print(f"\n[{lang}] Loading NRC VAD lexicon ({lang_name}) ...")
        try:
            lexicon = load_nrc_vad_v1(vad_dir, lang_name)
        except FileNotFoundError as e:
            print(f"  {e}")
            continue
        print(f"  Lexicon entries: {len(lexicon):,} tokens")

        filtered = load_jsonl(filt_path)
        target_ids = {r["utterance_id"] for r in filtered}
        sentiment_map = {r["utterance_id"]: r["sentiment_score"] for r in filtered}
        print(f"  Utterances:      {len(filtered):,}")

        # Fetch lemmas from original v4 JSONL
        v4_path = get_jsonl_path(cfg, lang)
        print(f"  Streaming v4 JSONL for lemmas: {v4_path} ...")
        lemma_map = _load_lemmas_from_v4(v4_path, target_ids, content_upos)
        n_with_lemmas = sum(1 for v in lemma_map.values() if v)
        print(f"  Utterances with lemmas: {n_with_lemmas:,}/{len(target_ids):,}")

        # Compute VAD per utterance
        rows = []
        for rec in filtered:
            uid    = rec["utterance_id"]
            lemmas = lemma_map.get(uid, [])
            vad    = _utterance_vad(lemmas, lexicon)
            rows.append({
                "utterance_id":    uid,
                "sentiment_score": rec["sentiment_score"],
                **vad,
            })

        df = pd.DataFrame(rows)
        valid = df.dropna(subset=["valence"])
        coverage = len(valid) / len(df) if len(df) > 0 else 0.0
        print(f"  VAD coverage (≥1 lemma matched): {len(valid):,}/{len(df):,} ({100*coverage:.1f}%)")

        # Correlations
        lang_corr: dict = {}
        for dim in ["valence", "arousal", "dominance"]:
            sub = df.dropna(subset=["sentiment_score", dim])
            if len(sub) < 10:
                lang_corr[dim] = {"r": None, "p": None, "n": len(sub)}
                continue
            if corr_method == "spearman":
                r, p = stats.spearmanr(sub["sentiment_score"], sub[dim])
            else:
                r, p = stats.pearsonr(sub["sentiment_score"], sub[dim])
            lang_corr[dim] = {"r": float(r), "p": float(p), "n": len(sub)}
            print(f"  sentiment ~ {dim}: r={r:.3f}, p={p:.4f}, n={len(sub):,}")

        correlations[lang] = lang_corr

        vad_out = idir / f"{lang}_vad.tsv"
        write_tsv(df, vad_out)
        print(f"  Written → {vad_out}")

    corr_out = rdir / "vad_correlations.json"
    with open(corr_out, "w") as f:
        json.dump(correlations, f, indent=2)
    print(f"\nWritten → {corr_out}")


if __name__ == "__main__":
    main()
