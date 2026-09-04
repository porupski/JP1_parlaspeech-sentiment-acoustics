# ============================================================
# Script:  extraction.py
# Release: 1.0
# Version: v1.00
# Purpose: Praat and speech-rate feature extraction from audio + word alignments
# ============================================================

import numpy as np
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Vowel sets per language — used for syllable counting
# ---------------------------------------------------------------------------
VOWELS = {
    "HR": set("aeiouAEIOUáéíóúÁÉÍÓÚàèìòùÀÈÌÒÙ"),
    "RS": set("aeiouAEIOUáéíóúÁÉÍÓÚàèìòùÀÈÌÒÙ"),
    "CZ": set("aeiouAEIOUáéíóúýÁÉÍÓÚÝůŮěĚ"),
    "PL": set("aeiouAEIOUáéíóúÁÉÍÓÚąęóĄĘÓ"),
    "SI": set("aeiouAEIOUáéíóúÁÉÍÓÚ"),
}


def count_syllables(word: str, lang: str) -> int:
    """Syllable count = vowel count (language-specific vowel set)."""
    vowels = VOWELS.get(lang, VOWELS["HR"])
    return sum(1 for ch in word if ch in vowels)


# ---------------------------------------------------------------------------
# Praat extraction (word-level median → utterance-level median)
# ---------------------------------------------------------------------------

def load_audio_segment(audio_path: str | Path,
                        start: float, end: float) -> parselmouth.Sound:
    """Load a segment of an audio file. start/end in seconds."""
    snd = parselmouth.Sound(str(audio_path))
    return snd.extract_part(from_time=start, to_time=end,
                             preserve_times=True, fade_length=0.0025)


def extract_f0_word(snd: parselmouth.Sound,
                     start: float, end: float,
                     pitch_floor: float = 75.0,
                     pitch_ceiling: float = 600.0) -> Optional[float]:
    """Median F0 over a word segment. Returns None if no voiced frames."""
    seg = snd.extract_part(from_time=start, to_time=end, preserve_times=True)
    if seg.duration < 0.02:
        return None
    try:
        pitch = seg.to_pitch(time_step=0.01,
                              pitch_floor=pitch_floor,
                              pitch_ceiling=pitch_ceiling)
        vals = pitch.selected_array["frequency"]
        voiced = vals[vals > 0]
        return float(np.median(voiced)) if len(voiced) > 0 else None
    except Exception:
        return None


def extract_intensity_word(snd: parselmouth.Sound,
                            start: float, end: float,
                            min_db: float = 50.0) -> Optional[float]:
    """Median intensity (dB) over a word segment."""
    seg = snd.extract_part(from_time=start, to_time=end, preserve_times=True)
    if seg.duration < 0.02:
        return None
    try:
        intensity = seg.to_intensity(time_step=0.01, minimum_pitch=min_db)
        vals = intensity.values[0]
        return float(np.median(vals[np.isfinite(vals)])) if len(vals) > 0 else None
    except Exception:
        return None


def extract_praat_utterance(audio_path: str | Path,
                             words_align: list[dict],
                             pitch_floor: float = 75.0,
                             pitch_ceiling: float = 600.0,
                             min_intensity_db: float = 50.0) -> dict:
    """
    Extract utterance-level F0 and intensity.
    Method: word-level median, then median across words (matches paper §3.3).

    words_align: list of {'word': str, 'start': float, 'end': float}
    Returns: {'f0_raw': float|None, 'intensity_raw': float|None}
    """
    if not words_align:
        return {"f0_raw": None, "intensity_raw": None}

    utt_start = words_align[0]["start"]
    utt_end = words_align[-1]["end"]

    try:
        snd = parselmouth.Sound(str(audio_path))
        # clip to utterance (small buffer)
        buf = 0.05
        snd_seg = snd.extract_part(
            from_time=max(0, utt_start - buf),
            to_time=min(snd.duration, utt_end + buf),
            preserve_times=True,
        )
    except Exception:
        return {"f0_raw": None, "intensity_raw": None}

    f0_words, int_words = [], []
    for w in words_align:
        if w["end"] - w["start"] < 0.02:
            continue
        f0 = extract_f0_word(snd_seg, w["start"], w["end"], pitch_floor, pitch_ceiling)
        intens = extract_intensity_word(snd_seg, w["start"], w["end"], min_intensity_db)
        if f0 is not None:
            f0_words.append(f0)
        if intens is not None:
            int_words.append(intens)

    return {
        "f0_raw": float(np.median(f0_words)) if f0_words else None,
        "intensity_raw": float(np.median(int_words)) if int_words else None,
    }


# ---------------------------------------------------------------------------
# Speech rate (from word alignment, no audio needed)
# ---------------------------------------------------------------------------

def extract_speechrate_utterance(words_align: list[dict], lang: str) -> dict:
    """
    Compute speech rate features from word-level alignments.

    Outputs:
        n_words: word count
        n_syllables: total syllable count
        duration_total: full utterance span (s)
        duration_speech: sum of word durations, no pauses (s)
        pause_ratio: (total - speech) / total
        speechrate_wps: words / duration_speech
        speechrate_sps: syllables / duration_speech
    """
    if not words_align:
        return {k: None for k in [
            "n_words", "n_syllables", "duration_total", "duration_speech",
            "pause_ratio", "speechrate_wps", "speechrate_sps"
        ]}

    words = [w["word"] for w in words_align]
    starts = [w["start"] for w in words_align]
    ends = [w["end"] for w in words_align]

    n_words = len(words)
    n_syllables = sum(count_syllables(w, lang) for w in words)
    duration_total = ends[-1] - starts[0]
    duration_speech = sum(e - s for s, e in zip(starts, ends))

    if duration_speech <= 0:
        return {
            "n_words": n_words, "n_syllables": n_syllables,
            "duration_total": duration_total, "duration_speech": duration_speech,
            "pause_ratio": None, "speechrate_wps": None, "speechrate_sps": None,
        }

    pause_ratio = (duration_total - duration_speech) / duration_total if duration_total > 0 else None
    speechrate_wps = n_words / duration_speech
    speechrate_sps = n_syllables / duration_speech if n_syllables > 0 else None

    return {
        "n_words": n_words,
        "n_syllables": n_syllables,
        "duration_total": duration_total,
        "duration_speech": duration_speech,
        "pause_ratio": pause_ratio,
        "speechrate_wps": speechrate_wps,
        "speechrate_sps": speechrate_sps,
    }
