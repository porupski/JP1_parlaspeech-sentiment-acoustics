#!/usr/bin/env python3
# ============================================================
# Script:  10_filter.py
# Release: 1.0
# Version: v1.00
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
from utils.data_utils import load_jsonl, write_jsonl, score_to_label


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--langs", nargs="+", default=None,
                   help="Languages to process (default: all from config)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats but do not write output")
    return p.parse_args()


def filter_language(records: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    """
    Two-pass filter:
      Pass 1: word count (10-40)
      Pass 2: speaker coverage (>= min_per_label across all 6 labels)
    Returns (filtered_records, stats_dict)
    """
    f = cfg["filter"]
    fields = cfg["jsonl_fields"]

    min_w = f["min_words"]
    max_w = f["max_words"]
    min_per_label = f["min_instances_per_label"]
    sent_labels = f["sentiment_labels"]

    fid = fields["utterance_id"]
    fspk = fields["speaker_id"]
    fsess = fields["session_id"]
    fwords = fields["words_align"]
    fscore = fields["sentiment_score"]
    flabel = fields["sentiment_label"]
    faudio = fields["audio_path"]
    fgender = fields.get("gender", "gender")

    # Pass 1: word count
    pass1 = []
    for rec in records:
        words = rec.get(fwords) or []
        n_words = len(words)
        if min_w <= n_words <= max_w:
            # Determine sentiment label (use existing or derive from score)
            label = rec.get(flabel)
            if label is None:
                score = rec.get(fscore)
                label = score_to_label(float(score)) if score is not None else None
            if label is None:
                continue
            pass1.append({
                "utterance_id": rec.get(fid),
                "speaker_id": rec.get(fspk),
                "session_id": rec.get(fsess),
                "sentiment_score": float(rec.get(fscore, 0.0)),
                "sentiment_label": label,
                "n_words": n_words,
                "words_align": words,
                "audio": rec.get(faudio),
                "gender": rec.get(fgender),
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
        print(f"[{lang}] After word filter:    {stats['after_word_filter']:,}")
        print(f"[{lang}] After speaker filter: {stats['after_speaker_filter']:,} "
              f"({stats['valid_speakers']} speakers)")

        if not args.dry_run:
            out_path = out_dir / f"{lang}_filtered.jsonl"
            write_jsonl(filtered, out_path)
            print(f"[{lang}] Written → {out_path}")


if __name__ == "__main__":
    main()
