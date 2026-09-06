#!/usr/bin/env python3
# ============================================================
# Script:  35_vad.py
# Release: 1.0
# Version: v1.20
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
#   5. Optionally save per-word VAD envelope NPZ (word_starts, word_ends,
#      word_valences, word_arousals, word_dominances — NaN where not covered).
#
# Notes on language proxies:
#   RS (Serbian): NRC v1 Serbian file is Cyrillic; ParlaSpeech RS is Latin.
#      Using Bosnian (Latin, mutually intelligible) as proxy. See config RS note.
#
# Input:   {intermediate_dir}/{lang}_filtered.jsonl  (utterance IDs + sentiment)
#          {data_root}/ParlaSpeech-{lang}.v4.0.patched.jsonl  (lemmas + words)
#          {nrc_vad_dir}/{LangName}-NRC-VAD-Lexicon.txt
# Output:  {intermediate_dir}/{lang}_vad.tsv
#          {intermediate_dir}/{lang}_vad_envelopes.npz  (if save_vad_envelopes)
#          {results_dir}/vad_correlations.json
#
# v1.20: Language-level parallelism via --workers (each language = one process).
#        Per-word VAD envelope NPZ: word_starts, word_ends, word_valences,
#        word_arousals, word_dominances (NaN where OOV or non-content UPOS).
#        Reads v4 words[] for timing alongside linguistic_annotation for lemmas.
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
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        f.readline()  # skip header
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


def _load_from_v4(jsonl_path: Path,
                  target_ids: set[str],
                  content_upos: set[str]) -> dict[str, dict]:
    """
    Stream v4 JSONL and extract lemmas + word timing for target utterance IDs.

    Returns dict: {utterance_id: {
        'lemmas': [lowercase content-word lemmas],
        'lemma_by_idx': {words_idx -> lemma},   # for envelope alignment
        'words': [{time_s, time_e, ...}],        # from v4 words[] field
    }}
    """
    result: dict[str, dict] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if len(result) >= len(target_ids):
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

            ling      = rec.get("linguistic_annotation", [])
            words_raw = rec.get("words", [])

            lemmas: list[str] = []
            lemma_by_idx: dict[int, str] = {}
            for tok in ling:
                if (tok.get("upos") in content_upos
                        and tok.get("lemma")
                        and tok.get("words_idx") is not None):
                    lm = tok["lemma"].lower()
                    lemmas.append(lm)
                    lemma_by_idx[tok["words_idx"]] = lm

            result[uid] = {
                "lemmas":       lemmas,
                "lemma_by_idx": lemma_by_idx,
                "words":        words_raw,
            }
    return result


def _vad_word_envelope(words: list[dict],
                       lemma_by_idx: dict[int, str],
                       lexicon: dict
                       ) -> tuple[np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-word VAD arrays aligned to words[] timing.
    NaN where word has no content-lemma or lemma not in lexicon.

    Returns (starts, ends, valences, arousals, dominances) all float32.
    """
    n = len(words)
    starts     = np.array([w.get("time_s", np.nan) for w in words], dtype=np.float32)
    ends       = np.array([w.get("time_e", np.nan) for w in words], dtype=np.float32)
    valences   = np.full(n, np.nan, dtype=np.float32)
    arousals   = np.full(n, np.nan, dtype=np.float32)
    dominances = np.full(n, np.nan, dtype=np.float32)
    for j, lemma in lemma_by_idx.items():
        if j < n and lemma in lexicon:
            entry = lexicon[lemma]
            valences[j]   = entry["valence"]
            arousals[j]   = entry["arousal"]
            dominances[j] = entry["dominance"]
    return starts, ends, valences, arousals, dominances


def _save_vad_envelopes_npz(path: Path, envelope_pairs: list) -> None:
    """
    Save per-word VAD envelope arrays to a single compressed NPZ.

    Load example:
        data = np.load('HR_vad_envelopes.npz', allow_pickle=True)
        idx = {uid: i for i, uid in enumerate(data['utterance_ids'])}
        val = data['word_valences'][idx['some_id']]  # shape (n_words,), NaN = not covered
    """
    if not envelope_pairs:
        return
    uids = [uid for uid, _ in envelope_pairs]
    arrays: dict = {"utterance_ids": np.array(uids, dtype=object)}
    for key, col_idx in [("word_starts", 0), ("word_ends", 1),
                          ("word_valences", 2), ("word_arousals", 3),
                          ("word_dominances", 4)]:
        ragged = []
        for _, env in envelope_pairs:
            ragged.append(env[col_idx] if env is not None
                          else np.array([], dtype=np.float32))
        arrays[key] = np.array(ragged, dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# Per-language worker (runs in subprocess when workers > 1)
# ---------------------------------------------------------------------------

def _process_language(args_tuple: tuple) -> tuple[str, dict, str]:
    """
    Process one language end-to-end: load lexicon, stream v4 JSONL,
    compute VAD + correlations, write TSV and envelope NPZ if requested.

    Returns (lang, correlations_dict, log_string).
    """
    (lang, lang_name, filt_path, v4_path, vad_dir,
     content_upos, corr_method, idir, save_envelopes) = args_tuple

    log: list[str] = []
    log.append(f"\n[{lang}] Loading NRC VAD lexicon ({lang_name}) ...")

    try:
        lexicon = load_nrc_vad_v1(Path(vad_dir), lang_name)
    except FileNotFoundError as e:
        log.append(f"  {e}")
        return lang, {}, "\n".join(log)

    log.append(f"  Lexicon entries: {len(lexicon):,} tokens")

    filtered = load_jsonl(Path(filt_path))
    target_ids = {r["utterance_id"] for r in filtered}
    log.append(f"  Utterances:      {len(filtered):,}")

    log.append(f"  Streaming v4 JSONL for lemmas: {v4_path} ...")
    v4_data = _load_from_v4(Path(v4_path), target_ids, content_upos)
    n_with_lemmas = sum(1 for v in v4_data.values() if v["lemmas"])
    log.append(f"  Utterances with lemmas: {n_with_lemmas:,}/{len(target_ids):,}")

    rows = []
    envelope_pairs = []
    for rec in filtered:
        uid  = rec["utterance_id"]
        data = v4_data.get(uid, {"lemmas": [], "lemma_by_idx": {}, "words": []})
        vad  = _utterance_vad(data["lemmas"], lexicon)
        rows.append({
            "utterance_id":    uid,
            "sentiment_score": rec["sentiment_score"],
            **vad,
        })
        if save_envelopes:
            env = _vad_word_envelope(data["words"], data["lemma_by_idx"], lexicon)
            envelope_pairs.append((uid, env))

    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["valence"])
    coverage = len(valid) / len(df) if len(df) > 0 else 0.0
    log.append(f"  VAD coverage (≥1 lemma matched): {len(valid):,}/{len(df):,} "
               f"({100*coverage:.1f}%)")

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
        log.append(f"  sentiment ~ {dim}: r={r:.3f}, p={p:.4f}, n={len(sub):,}")

    vad_out = Path(idir) / f"{lang}_vad.tsv"
    write_tsv(df, vad_out)
    log.append(f"  Written → {vad_out}")

    if save_envelopes and envelope_pairs:
        npz_out = Path(idir) / f"{lang}_vad_envelopes.npz"
        _save_vad_envelopes_npz(npz_out, envelope_pairs)
        n_covered = sum(1 for _, env in envelope_pairs
                        if env is not None and np.any(~np.isnan(env[2])))
        log.append(f"  Envelope NPZ → {npz_out}  ({n_covered:,} utterances with ≥1 word covered)")

    return lang, lang_corr, "\n".join(log)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None)
    p.add_argument("--workers", type=int, default=5,
                   help="Parallel worker processes (one per language; capped at n_langs)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if not cfg["analysis"]["enable_text_vad"]:
        print("Text VAD disabled (enable_text_vad = false). Exiting.")
        return

    vad_cfg        = cfg["vad"]
    vad_dir        = Path(vad_cfg["nrc_vad_dir"])
    lang_codes     = vad_cfg.get("nrc_lang_codes", {})
    content_upos   = set(vad_cfg.get("vad_content_upos", ["NOUN", "VERB", "ADJ", "ADV"]))
    corr_method    = vad_cfg.get("vad_corr_method", "spearman")
    save_envelopes = vad_cfg.get("save_vad_envelopes", False)

    if not vad_dir.exists():
        print(f"NRC VAD directory not found: {vad_dir}")
        print("Run: bash 0_env/get_nrc_vad.sh")
        return

    langs = args.langs or cfg["languages"]
    idir  = get_intermediate_dir(cfg)
    rdir  = get_results_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)

    work_items = []
    for lang in langs:
        filt_path = idir / f"{lang}_filtered.jsonl"
        if not filt_path.exists():
            print(f"[{lang}] {filt_path} not found. Run 10_filter.py first.")
            continue
        lang_name = lang_codes.get(lang)
        if not lang_name:
            print(f"[{lang}] No nrc_lang_codes entry. Skipping.")
            continue
        work_items.append((
            lang, lang_name, str(filt_path), str(get_jsonl_path(cfg, lang)),
            str(vad_dir), content_upos, corr_method, str(idir), save_envelopes,
        ))

    if not work_items:
        return

    n_workers = min(args.workers, len(work_items))
    correlations: dict = {}

    if n_workers <= 1 or len(work_items) == 1:
        for item in work_items:
            lang, corr, log = _process_language(item)
            print(log)
            correlations[lang] = corr
    else:
        print(f"Running {len(work_items)} languages in parallel "
              f"({n_workers} workers) ...")
        futures: dict = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for item in work_items:
                f = pool.submit(_process_language, item)
                futures[f] = item[0]
            for f in as_completed(futures):
                lang, corr, log = f.result()
                print(log, flush=True)
                correlations[lang] = corr

    corr_out = rdir / "vad_correlations.json"
    with open(corr_out, "w") as f:
        json.dump(correlations, f, indent=2)
    print(f"\nWritten → {corr_out}")


if __name__ == "__main__":
    main()
