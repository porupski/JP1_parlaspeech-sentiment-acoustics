#!/usr/bin/env python3
# ============================================================
# Script:  10_filter.py
# Release: 1.0
# Version: v1.01
# Purpose: Filter ParlaSpeech v4 JSONL by word count and speaker
#          sentiment coverage. Outputs one compact JSONL per language.
#
# Input:   {data_root}/ParlaSpeech-{LANG}.v4.0.patched.jsonl
# Output:  {intermediate_dir}/{lang}_filtered.jsonl
#
# Filtering rules (from paper §3.1 and config.json):
#   - Utterances: 10-40 words
#   - Speakers: >= min_instances_per_label per each of 6 sentiment categories
# ============================================================

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_jsonl_path, get_intermediate_dir, get_field
from utils.data_utils import load_jsonl, write_jsonl, score_to_label, get_nested


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None,
                   help="Languages to process (default: all from config)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats but do not write output")
    return p.parse_args()


def _normalize_words_align(words_raw: list, text: str, wa_fields: dict) -> list:
    """Convert JSONL words_align entries to internal format {word, start, end}."""
    f_s = wa_fields.get("start", "time_s")
    f_e = wa_fields.get("end", "time_e")
    f_cs = wa_fields.get("char_start", "char_s")
    f_ce = wa_fields.get("char_end", "char_e")
    result = []
    for w in (words_raw or []):
        cs = w.get(f_cs)
        ce = w.get(f_ce)
        word = text[cs:ce] if (text and cs is not None and ce is not None) else ""
        result.append({"word": word, "start": w.get(f_s), "end": w.get(f_e)})
    return result


def filter_language(records: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    """
    Two-pass filter:
      Pass 1: word count (10-40)
      Pass 2: speaker coverage (>= min_per_label across all 6 labels)
    Returns (filtered_records, stats_dict)
    """
    f = cfg["filter"]
    fields = cfg["jsonl_fields"]
    wa_fields = cfg.get("words_align_fields", {})

    min_w = f["min_words"]
    max_w = f["max_words"]
    min_per_label = f["min_instances_per_label"]
    sent_labels = f["sentiment_labels"]

    fid = fields["utterance_id"]
    fspk = fields["speaker_id"]
    fsess = fields["session_id"]
    ftext = fields["text"]
    fwords = fields["words_align"]
    fscore = fields["sentiment_score"]
    flabel = fields["sentiment_label"]
    faudio = fields["audio_path"]
    fgender = fields.get("gender", "gender")

    # Pass 1: word count
    pass1 = []
    n_no_words = 0
    n_no_label = 0
    for rec in records:
        words_raw = get_nested(rec, fwords) or []
        text = get_nested(rec, ftext, "")
        words = _normalize_words_align(words_raw, text, wa_fields)
        n_words = len(words)
        if not (min_w <= n_words <= max_w):
            n_no_words += 1
            continue
        label = get_nested(rec, flabel)
        if label is None:
            score = get_nested(rec, fscore)
            label = score_to_label(float(score)) if score is not None else None
        if label is None:
            n_no_label += 1
            continue
        score_val = get_nested(rec, fscore)
        pass1.append({
                "utterance_id": get_nested(rec, fid),
                "speaker_id": get_nested(rec, fspk),
                "session_id": get_nested(rec, fsess),
                "sentiment_score": float(score_val) if score_val is not None else 0.0,
                "sentiment_label": label,
                "n_words": n_words,
                "words_align": words,
                "audio": get_nested(rec, faudio),
                "gender": get_nested(rec, fgender),
                "silent_pauses": rec.get("silent_pauses") or [],
                "filled_pauses": rec.get("filled_pauses") or [],
            })

    # Pass 2: speaker coverage
    spk_label_counts = defaultdict(lambda: defaultdict(int))
    for rec in pass1:
        spk_label_counts[rec["speaker_id"]][rec["sentiment_label"]] += 1

    valid_speakers = {
        spk for spk, counts in spk_label_counts.items()
        if all(counts.get(lbl, 0) >= min_per_label for lbl in sent_labels)
    }

    filtered = [r for r in pass1 if r["speaker_id"] in valid_speakers]

    stats = {
        "raw_utterances": len(records),
        "after_word_filter": len(pass1),
        "after_speaker_filter": len(filtered),
        "valid_speakers": len(valid_speakers),
        "total_speakers_pass1": len(spk_label_counts),
        "dropped_word_count": n_no_words,
        "dropped_no_label": n_no_label,
    }
    return filtered, stats


def main():
    args = parse_args()
    cfg = load_config(args.config)
    langs = args.langs or cfg["languages"]
    out_dir = get_intermediate_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        jsonl_path = get_jsonl_path(cfg, lang)
        print(f"\n[{lang}] Loading {jsonl_path} ...")
        records = load_jsonl(jsonl_path)
        print(f"[{lang}] {len(records):,} raw utterances")

        filtered, stats = filter_language(records, cfg)
        print(f"[{lang}] After word filter:    {stats['after_word_filter']:,}"
              f"  (dropped word-count: {stats['dropped_word_count']:,}"
              f"  no-label: {stats['dropped_no_label']:,})")
        print(f"[{lang}] After speaker filter: {stats['after_speaker_filter']:,} "
              f"({stats['valid_speakers']} speakers)")

        if not args.dry_run:
            out_path = out_dir / f"{lang}_filtered.jsonl"
            write_jsonl(filtered, out_path)
            print(f"[{lang}] Written → {out_path}")


if __name__ == "__main__":
    main()
