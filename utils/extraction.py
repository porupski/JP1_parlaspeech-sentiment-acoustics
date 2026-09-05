# ============================================================
# Script:  extraction.py
# Release: 1.0
# Version: v1.10
# Purpose: Praat and speech-rate feature extraction from audio + word alignments.
#
# v1.10 changes:
#   - Added F1/F2/F3, HNR (word-level), jitter/shimmer/HNR_utt (utterance-level)
#   - Analysis objects (Pitch, Intensity, Formant, Harmonicity) computed once per
#     utterance and sampled per word — avoids repeated object creation overhead
#   - silent_pauses tier used to mask PointProcess (jitter/shimmer more accurate)
#   - save_envelopes flag: returns raw F0/intensity tracks + per-word arrays
#   - Speech rate: pause breakdown using silent_pauses + filled_pauses tiers
#   - pause_ratio renamed to pause_ratio_all; pause_ratio_silent is the new main metric
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
    vowels = VOWELS.get(lang, VOWELS["HR"])
    return sum(1 for ch in word if ch in vowels)


# ---------------------------------------------------------------------------
# Internal helpers — operate on pre-computed utterance-level Praat objects
# ---------------------------------------------------------------------------

def _f0_word_stats(pitch_obj: parselmouth.Pitch,
                   start: float, end: float) -> tuple[Optional[float], Optional[float]]:
    """Mean and median voiced F0 within [start, end] from a pre-computed Pitch object."""
    times = pitch_obj.xs()
    freqs = pitch_obj.selected_array["frequency"]
    mask = (times >= start) & (times <= end)
    voiced = freqs[mask & (freqs > 0)]
    if len(voiced) == 0:
        return None, None
    return float(np.mean(voiced)), float(np.median(voiced))


def _intensity_word_stats(intensity_obj: parselmouth.Intensity,
                           start: float, end: float) -> tuple[Optional[float], Optional[float]]:
    """Mean and median intensity within [start, end] from a pre-computed Intensity object."""
    times = intensity_obj.xs()
    vals = intensity_obj.values[0]
    mask = (times >= start) & (times <= end)
    finite = vals[mask & np.isfinite(vals)]
    if len(finite) == 0:
        return None, None
    return float(np.mean(finite)), float(np.median(finite))


def _formants_word(formant_obj: parselmouth.Formant,
                   start: float, end: float,
                   n_samples: int = 5) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    F1/F2/F3 median at N evenly spaced sample points within [start, end].
    Validity ranges: F1 200-1200 Hz, F2 600-3500 Hz, F3 1500-5000 Hz.
    """
    t_pts = np.linspace(start, end, n_samples)
    f1s, f2s, f3s = [], [], []
    for t in t_pts:
        v1 = formant_obj.get_value_at_time(1, t)
        v2 = formant_obj.get_value_at_time(2, t)
        v3 = formant_obj.get_value_at_time(3, t)
        if v1 and 200.0 < v1 < 1200.0:
            f1s.append(v1)
        if v2 and 600.0 < v2 < 3500.0:
            f2s.append(v2)
        if v3 and 1500.0 < v3 < 5000.0:
            f3s.append(v3)
    f1 = float(np.median(f1s)) if f1s else None
    f2 = float(np.median(f2s)) if f2s else None
    f3 = float(np.median(f3s)) if f3s else None
    return f1, f2, f3


def _hnr_word(harmony_obj: parselmouth.Harmonicity,
              start: float, end: float) -> Optional[float]:
    """Median HNR within [start, end] from a pre-computed Harmonicity object.
    Praat marks unvoiced frames as -200 dB; those are excluded."""
    times = harmony_obj.xs()
    vals = harmony_obj.values[0]
    mask = (times >= start) & (times <= end)
    valid = vals[mask & (vals > -200.0)]
    return float(np.median(valid)) if len(valid) > 0 else None


# ---------------------------------------------------------------------------
# Speech intervals — used for voice quality extraction
# ---------------------------------------------------------------------------

def _build_speech_intervals(words_align: list[dict],
                              silent_pauses: Optional[list[dict]],
                              gap_threshold: float = 0.05) -> list[tuple[float, float]]:
    """
    Speech-only intervals = merged word spans minus confirmed silent_pauses.
    Returns sorted list of (start, end) tuples, each >= 20 ms.
    """
    ivs = [(w["start"], w["end"]) for w in words_align
           if w["end"] > w["start"] + 0.01]
    if not ivs:
        return []

    ivs.sort()
    merged: list[list[float]] = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1] + gap_threshold:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    speech = [(s, e) for s, e in merged]

    if not silent_pauses:
        return speech

    sp_ivs = [(sp["time_s"], sp["time_e"]) for sp in silent_pauses]
    result: list[tuple[float, float]] = []
    for seg_s, seg_e in speech:
        parts: list[tuple[float, float]] = [(seg_s, seg_e)]
        for sp_s, sp_e in sp_ivs:
            new: list[tuple[float, float]] = []
            for ps, pe in parts:
                if sp_e <= ps or sp_s >= pe:
                    new.append((ps, pe))
                else:
                    if ps < sp_s:
                        new.append((ps, sp_s))
                    if sp_e < pe:
                        new.append((sp_e, pe))
            parts = new
        result.extend(parts)

    return [(s, e) for s, e in result if e - s >= 0.02]


# ---------------------------------------------------------------------------
# Utterance-level voice quality (jitter, shimmer, HNR)
# ---------------------------------------------------------------------------

_VQ_NULL = {
    "jitter_local": None, "jitter_rap": None,
    "shimmer_local": None, "shimmer_local_db": None, "shimmer_apq11": None,
    "hnr_utt": None,
}


def _voice_quality_utterance(snd_seg: parselmouth.Sound,
                               words_align: list[dict],
                               silent_pauses: Optional[list[dict]],
                               pitch_floor: float,
                               pitch_ceiling: float) -> dict:
    """
    Jitter/shimmer/HNR computed via PointProcess on speech-only audio.
    Speech-only = word intervals merged, confirmed silent_pauses subtracted.
    """
    ivs = _build_speech_intervals(words_align, silent_pauses)
    if not ivs:
        return dict(_VQ_NULL)

    try:
        if len(ivs) == 1:
            s, e = ivs[0]
            speech = snd_seg.extract_part(from_time=s, to_time=e, preserve_times=False)
        else:
            segs = [snd_seg.extract_part(from_time=s, to_time=e, preserve_times=False)
                    for s, e in ivs]
            speech = call(segs, "Concatenate")
    except Exception:
        return dict(_VQ_NULL)

    if speech.duration < 0.1:
        return dict(_VQ_NULL)

    try:
        pp = call(speech, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    except Exception:
        return dict(_VQ_NULL)

    res: dict = {}
    j_args = (0, 0, 0.0001, 0.02, 1.3)
    s_args = (0, 0, 0.0001, 0.02, 1.3, 1.6)

    for name, obj, cmd, args in [
        ("jitter_local",    pp,             "Get jitter (local)",    j_args),
        ("jitter_rap",      pp,             "Get jitter (rap)",      j_args),
        ("shimmer_local",   [speech, pp],   "Get shimmer (local)",   s_args),
        ("shimmer_local_db",[speech, pp],   "Get shimmer (local_dB)",s_args),
        ("shimmer_apq11",   [speech, pp],   "Get shimmer (apq11)",   s_args),
    ]:
        try:
            res[name] = float(call(obj, cmd, *args))
        except Exception:
            res[name] = None

    try:
        harm = speech.to_harmonicity_cc(time_step=0.01, minimum_pitch=pitch_floor,
                                         silence_threshold=0.1, periods_per_window=1.0)
        valid = harm.values[0]
        valid = valid[valid > -200.0]
        res["hnr_utt"] = float(np.median(valid)) if len(valid) > 0 else None
    except Exception:
        res["hnr_utt"] = None

    return res


# ---------------------------------------------------------------------------
# Main utterance extraction
# ---------------------------------------------------------------------------

_PRAAT_NULL = {
    "f0_raw": None, "intensity_raw": None,
    "f1_median": None, "f2_median": None, "f3_median": None,
    "hnr_median": None,
    "jitter_local": None, "jitter_rap": None,
    "shimmer_local": None, "shimmer_local_db": None, "shimmer_apq11": None,
    "hnr_utt": None,
}


def extract_praat_utterance(audio_path: "str | Path",
                             words_align: list[dict],
                             silent_pauses: Optional[list[dict]] = None,
                             pitch_floor: float = 75.0,
                             pitch_ceiling: float = 600.0,
                             min_intensity_db: float = 50.0,
                             formant_n_samples: int = 5,
                             formant_max_hz: float = 5500.0,
                             save_envelopes: bool = False) -> dict:
    """
    Extract utterance-level acoustic features via Praat.

    Method (paper §3.3): Pitch, Intensity, Formant, Harmonicity objects computed
    once per utterance; sampled per word → word-level median → utterance median.
    Jitter/shimmer/HNR_utt use PointProcess on speech-only intervals
    (words_align merged, silent_pauses subtracted).

    words_align: list of {'word': str, 'start': float, 'end': float}
    silent_pauses: list of {'time_s': float, 'time_e': float, ...} from v4 JSONL

    Returns scalar features dict. If save_envelopes=True, also returns
    '_envelopes' key with raw tracks and per-word arrays for plotting.
    """
    null = dict(_PRAAT_NULL)
    if save_envelopes:
        null["_envelopes"] = None

    if not words_align:
        return null

    utt_start = words_align[0]["start"]
    utt_end = words_align[-1]["end"]
    buf = 0.05

    try:
        snd = parselmouth.Sound(str(audio_path))
        snd_seg = snd.extract_part(
            from_time=max(0.0, utt_start - buf),
            to_time=min(snd.duration, utt_end + buf),
            preserve_times=True,
        )
    except Exception:
        return null

    # Pre-compute analysis objects once (efficient: sample per word instead of recomputing)
    pitch_obj = intensity_obj = formant_obj = harmony_obj = None
    try:
        pitch_obj = snd_seg.to_pitch(time_step=0.01,
                                      pitch_floor=pitch_floor,
                                      pitch_ceiling=pitch_ceiling)
    except Exception:
        pass
    try:
        intensity_obj = snd_seg.to_intensity(time_step=0.01,
                                              minimum_pitch=min_intensity_db)
    except Exception:
        pass
    try:
        formant_obj = snd_seg.to_formant_burg(time_step=0.01,
                                               max_number_of_formants=5.0,
                                               maximum_formant=formant_max_hz,
                                               window_length=0.025,
                                               pre_emphasis_from=50.0)
    except Exception:
        pass
    try:
        harmony_obj = snd_seg.to_harmonicity_cc(time_step=0.01,
                                                  minimum_pitch=pitch_floor,
                                                  silence_threshold=0.1,
                                                  periods_per_window=1.0)
    except Exception:
        pass

    # Per-word feature extraction
    f0_word_means:    list[Optional[float]] = []
    f0_word_medians:  list[Optional[float]] = []
    int_word_means:   list[Optional[float]] = []
    int_word_medians: list[Optional[float]] = []
    f1_words: list[Optional[float]] = []
    f2_words: list[Optional[float]] = []
    f3_words: list[Optional[float]] = []
    hnr_words: list[Optional[float]] = []
    word_starts_out: list[float] = []
    word_ends_out:   list[float] = []

    for w in words_align:
        ws, we = w["start"], w["end"]
        if we - ws < 0.02:
            continue
        word_starts_out.append(ws)
        word_ends_out.append(we)

        if pitch_obj is not None:
            fm, fmed = _f0_word_stats(pitch_obj, ws, we)
        else:
            fm, fmed = None, None
        f0_word_means.append(fm)
        f0_word_medians.append(fmed)

        if intensity_obj is not None:
            im, imed = _intensity_word_stats(intensity_obj, ws, we)
        else:
            im, imed = None, None
        int_word_means.append(im)
        int_word_medians.append(imed)

        if formant_obj is not None:
            f1, f2, f3 = _formants_word(formant_obj, ws, we, formant_n_samples)
        else:
            f1, f2, f3 = None, None, None
        f1_words.append(f1)
        f2_words.append(f2)
        f3_words.append(f3)

        if harmony_obj is not None:
            hnr = _hnr_word(harmony_obj, ws, we)
        else:
            hnr = None
        hnr_words.append(hnr)

    def _med(lst: list) -> Optional[float]:
        vals = [x for x in lst if x is not None]
        return float(np.median(vals)) if vals else None

    scalars = {
        "f0_raw":        _med(f0_word_medians),
        "intensity_raw": _med(int_word_medians),
        "f1_median":     _med(f1_words),
        "f2_median":     _med(f2_words),
        "f3_median":     _med(f3_words),
        "hnr_median":    _med(hnr_words),
    }

    vq = _voice_quality_utterance(snd_seg, words_align, silent_pauses,
                                   pitch_floor, pitch_ceiling)
    scalars.update(vq)

    if not save_envelopes:
        return scalars

    f0_times_raw = pitch_obj.xs().tolist() if pitch_obj is not None else []
    f0_vals_raw  = pitch_obj.selected_array["frequency"].tolist() if pitch_obj is not None else []
    int_times_raw = intensity_obj.xs().tolist() if intensity_obj is not None else []
    int_vals_raw  = intensity_obj.values[0].tolist() if intensity_obj is not None else []

    scalars["_envelopes"] = {
        "f0_times":             f0_times_raw,
        "f0_values":            f0_vals_raw,
        "intensity_times":      int_times_raw,
        "intensity_values":     int_vals_raw,
        "word_starts":          word_starts_out,
        "word_ends":            word_ends_out,
        "f0_word_mean":         f0_word_means,
        "f0_word_median":       f0_word_medians,
        "intensity_word_mean":  int_word_means,
        "intensity_word_median":int_word_medians,
        "f1_word_median":       f1_words,
        "f2_word_median":       f2_words,
        "f3_word_median":       f3_words,
    }

    return scalars


# ---------------------------------------------------------------------------
# Speech rate (from word alignment + pause tiers, no audio needed)
# ---------------------------------------------------------------------------

def extract_speechrate_utterance(words_align: list[dict],
                                  lang: str,
                                  silent_pauses: Optional[list[dict]] = None,
                                  filled_pauses: Optional[list[dict]] = None) -> dict:
    """
    Compute speech rate and pause features from word-level alignments.

    words_align entries: {'word': str, 'start': float, 'end': float}
    silent_pauses: list of v4 silent_pauses tier entries
    filled_pauses: list of v4 filled_pauses tier entries

    Outputs:
        n_words, n_syllables
        duration_total:     full utterance span (last_word_end - first_word_start)
        duration_speech:    sum of word durations
        pause_ratio_all:    (total - speech) / total  [old metric, kept for comparison]
        pause_ratio_silent: dur_silent / total         [new main metric]
        n_silent_pauses, dur_silent
        n_filled_pauses, dur_filled
        speechrate_wps:     words / duration_speech
        speechrate_sps:     syllables / duration_speech
    """
    _null_keys = [
        "n_words", "n_syllables", "duration_total", "duration_speech",
        "pause_ratio_all", "pause_ratio_silent",
        "n_silent_pauses", "dur_silent",
        "n_filled_pauses", "dur_filled",
        "speechrate_wps", "speechrate_sps",
    ]
    if not words_align:
        return {k: None for k in _null_keys}

    words       = [w["word"] for w in words_align]
    starts      = [w["start"] for w in words_align]
    ends        = [w["end"] for w in words_align]

    n_words      = len(words)
    n_syllables  = sum(count_syllables(w, lang) for w in words)
    duration_total   = ends[-1] - starts[0]
    duration_speech  = sum(e - s for s, e in zip(starts, ends))

    # Pause breakdown from v4 tiers
    sp = silent_pauses or []
    fp = filled_pauses or []

    n_silent  = len(sp)
    dur_silent = sum(
        s.get("duration", s.get("time_e", 0.0) - s.get("time_s", 0.0))
        for s in sp
    )
    n_filled  = len(fp)
    dur_filled = sum(f.get("time_e", 0.0) - f.get("time_s", 0.0) for f in fp)

    if duration_total <= 0 or duration_speech <= 0:
        return {
            "n_words": n_words, "n_syllables": n_syllables,
            "duration_total": duration_total, "duration_speech": duration_speech,
            "pause_ratio_all": None, "pause_ratio_silent": None,
            "n_silent_pauses": n_silent, "dur_silent": dur_silent,
            "n_filled_pauses": n_filled, "dur_filled": dur_filled,
            "speechrate_wps": None, "speechrate_sps": None,
        }

    pause_ratio_all    = (duration_total - duration_speech) / duration_total
    pause_ratio_silent = dur_silent / duration_total
    speechrate_wps     = n_words / duration_speech
    speechrate_sps     = n_syllables / duration_speech if n_syllables > 0 else None

    return {
        "n_words":           n_words,
        "n_syllables":       n_syllables,
        "duration_total":    duration_total,
        "duration_speech":   duration_speech,
        "pause_ratio_all":   pause_ratio_all,
        "pause_ratio_silent":pause_ratio_silent,
        "n_silent_pauses":   n_silent,
        "dur_silent":        dur_silent,
        "n_filled_pauses":   n_filled,
        "dur_filled":        dur_filled,
        "speechrate_wps":    speechrate_wps,
        "speechrate_sps":    speechrate_sps,
    }
