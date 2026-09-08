#!/usr/bin/env python3
# ============================================================
# Script:  50_explore.py  (Jupyter-compatible notebook)
# Release: 1.0
# Version: v1.08
# Purpose: Debug and exploration for ParlaSpeech sentiment-acoustics.
#          Envelope viewer, speechrate+transcript, general replotters,
#          Praat vs OpenSMILE comparison, per-language anomaly inspector.
#
# Usage:   Open in VS Code with Jupyter extension (cells = # %% blocks)
#          Convert: jupytext --to notebook 5_debug/50_explore.py --output 5_debug/50_explore.ipynb
#          Run from: JP1_parlaspeech-sentiment-acoustics/ directory (kernel CWD irrelevant — paths are absolute)
#
# v1.08: New Cell 1b — per-lang coverage table (VAD % word-level + utt-level, F0/Int/SR
#        validity %). New Cell 6c — clean editorial trend plots (per-lang and GLOBAL,
#        F0 per-speaker z-scored + M/F splits, thick orange line, no overlays). New
#        Cell 10 — reads results/*.json (h1/h2/h3/numbers/vad) and prints a summary.
#        All per-lang & GLOBAL metrics also collected into `results.json` (dumped at
#        exit) so the log is verbose and the JSON is complete.
# v1.07: LANGS list drives all single-lang cells (previously LANG='SI' only ran SI).
#        Every plt.show() replaced with save_fig(descriptive_name) so headless PNGs
#        have meaningful filenames (e.g. cell02_envelope_SI_example03_<uid>.png).
#        Cell 8 gains per-language + GLOBAL merged Spearman ρ table for VAD.
#        Helpers extracted to 5_debug/explore_utils.py (FigSaver, logprint, ...).
#        Every numeric that appears on a plot now also goes to stdout via logprint().
# v1.06: --headless / --save-figs flags. In headless mode: matplotlib=Agg,
#        plt.show → savefig+close, stdout tee'd to logs/explore_<ts>/run.log.
#        Cell 6b gains SPLIT_BY_GENDER toggle (critical for F0 — M/F ranges
#        differ by ~100Hz; pooling raw hides true sentiment effect).
# v1.05: Cell 2 gains silent+filled pause overlays; Cell 2b prints VAD % coverage;
#        Cell 3 word-timeline gets color-scheme caption + filled-pause spans;
#        New Cell 6b — global trends (equal-language vs weighted-speaker) with
#        divergence summary. Topic cell now prefers `topic` column from filtered
#        JSONL (10_filter.py v1.02+), falls back to raw v4 only if missing.
# v1.04: Fix Cell 2/2b/5 hang — cache NpzFile arrays into locals. NpzFile.__getitem__
#        re-decompresses+unpickles the whole array per call; `data[key]` inside a loop
#        was doing N full disk-to-memory hits. Fix Cell 7 heatmap geometry (was tall/narrow).
#        Fix Cell 9 NameError (__JSONL_DIR → _JSONL_DIR).
# v1.03: TEST_RUN cap now covers every cell — added to Cell 2 (praat NPZ uids),
#        Cell 2b (VAD NPZ uids + VAD TSV), Cell 5 (LLD NPZ uids + osmile TSV),
#        Cell 7 (praat/osmile TSVs), VAD-corr cell, and topic-ANOVA cell.
# v1.02: idir/rdir now absolute (_repo_root-anchored) — no more kernel CWD dependency.
#        TEST_RUN=True initial impl (Setup, Cell 3, Cell 4, Cell 6 only).
#        Fixed VOWELS in Cell 3 (removed consonants HR/RS/SI). Version v1.01 skipped.
# v1.00: Initial notebook.
# ============================================================

# %% [markdown]
# # ParlaSpeech Sentiment-Acoustics — Exploration Notebook
#
# Cells can be run independently after the Setup cell.
# Adjust `LANG`, `SEED`, `N_EXAMPLES`, `REROLL` in Setup.

# %% Setup — run this first
import sys
import json
import os
import random
import warnings
import argparse
from datetime import datetime as _dt
from pathlib import Path

# --- CLI args (only meaningful when run as a script; ignored in Jupyter) ---
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--headless", action="store_true",
                 help="No display; save figs and tee stdout to logs/explore_<ts>/")
_ap.add_argument("--save-figs", default=None,
                 help="Directory for figures (implies --headless)")
_args_ns, _ = _ap.parse_known_args()
HEADLESS = _args_ns.headless or _args_ns.save_figs is not None

if HEADLESS:
    import matplotlib
    matplotlib.use("Agg", force=True)   # must precede pyplot import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

try:
    # Running as a script: __file__ is defined
    _repo_root = Path(__file__).resolve().parent.parent
except NameError:
    # Running as a notebook cell: walk up from CWD to find project root
    _cwd = Path(".").resolve()
    _repo_root = _cwd.parent if _cwd.name == "5_debug" else _cwd
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "5_debug"))   # for explore_utils import
os.chdir(_repo_root)   # make relative paths (data/intermediate/…) resolve from repo root

# --- Headless setup: fig-dir + stdout tee to log file ---
FIGS_DIR = None
if HEADLESS:
    FIGS_DIR = (Path(_args_ns.save_figs).resolve() if _args_ns.save_figs
                else (_repo_root / "logs" / f"explore_{_dt.now().strftime('%Y%m%d_%H%M%S')}").resolve())
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    class _Tee:
        def __init__(self, *streams): self._s = streams
        def write(self, s):
            for st in self._s:
                st.write(s)
            return len(s)
        def flush(self):
            for st in self._s: st.flush()
    _log_fh = open(FIGS_DIR / "run.log", "w", buffering=1, encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    print(f"[HEADLESS] Started {_dt.now().isoformat(timespec='seconds')}")
    print(f"[HEADLESS] Figures → {FIGS_DIR}")
    print(f"[HEADLESS] Log     → {FIGS_DIR / 'run.log'}")

from utils.config_loader import load_config, get_intermediate_dir, get_results_dir
from utils.data_utils import load_jsonl
from explore_utils import FigSaver, logprint, zscore_per_speaker, \
    binned_speaker_mean, clean_trend_plot, spearman_report

save_fig = FigSaver(FIGS_DIR, HEADLESS)   # every plot: save_fig("descriptive_name")

# --- Collect ALL computed numbers here; dumped at exit as results.json ---
import atexit
results_json: dict = {
    "notebook_version": "1.08",
    "started": _dt.now().isoformat(timespec="seconds"),
    "test_run": bool(HEADLESS and _args_ns is not None) and False,  # set below after LANGS
}
def _dump_results_json():
    _dst = (FIGS_DIR / "results.json") if HEADLESS else (_repo_root / "logs" / "explore_last_run_results.json")
    _dst.parent.mkdir(parents=True, exist_ok=True)
    with open(_dst, "w", encoding="utf-8") as _f:
        json.dump(results_json, _f, indent=2, default=str, ensure_ascii=False)
    print(f"[results.json] Wrote {_dst}")
atexit.register(_dump_results_json)

# ── Configure here ──────────────────────────────────────────────────────────
# LANGS drives every per-language loop. Single-lang cells (envelope viewers,
# transcript, praat vs osmile, VAD) iterate through LANGS. If you only want
# one language, put one code in the list. All-lang cells use ALL_LANGS.
LANGS      = ["SI"]      # e.g. ["HR"] or ["HR","CZ","PL","RS","SI"]
LANG       = LANGS[0]                        # back-compat alias for single-lang code paths
ALL_LANGS  = ["HR", "CZ", "PL", "RS", "SI"]  # used by global-trend and cross-lang cells
SEED       = 42      # reproducible sample; ignored when REROLL=True
N_EXAMPLES = 6       # utterances to display per envelope cell (per language)
REROLL     = False   # True = new random sample each run
TEST_RUN   = False   # True = cap every data load to TEST_RUN_N rows/records
TEST_RUN_N = 1_000
# ────────────────────────────────────────────────────────────────────────────

rng = random.Random(None if REROLL else SEED)

cfg   = load_config()  # auto-locates config.json via utils/__file__, not CWD
# Always absolute — idir/rdir are correct regardless of kernel CWD
idir  = (_repo_root / cfg["paths"]["intermediate_dir"]).resolve()
rdir  = (_repo_root / cfg["paths"]["results_dir"]).resolve()

PALETTE = {
    "HR": "#E63946", "CZ": "#2A9D8F", "PL": "#E9C46A",
    "RS": "#264653", "SI": "#A8DADC",
}

results_json.update({"langs": LANGS, "all_langs": ALL_LANGS, "test_run": TEST_RUN,
                     "test_run_n": TEST_RUN_N if TEST_RUN else None,
                     "n_examples": N_EXAMPLES, "seed": SEED})

# Load features TSV for LANG (if present)
feats_path = idir / f"{LANG}_features.tsv"
_nrows = TEST_RUN_N if TEST_RUN else None
df_feats = pd.read_csv(feats_path, sep="\t", nrows=_nrows) if feats_path.exists() else pd.DataFrame()
if df_feats.empty:
    print(f"[WARN] {feats_path} not found — feature columns unavailable in some cells.")
else:
    _cap = f" (TEST_RUN cap={TEST_RUN_N:,})" if TEST_RUN else ""
    print(f"Loaded {len(df_feats):,} utterances from {feats_path.name}{_cap}")
    print(f"  Columns: {list(df_feats.columns[:12])} ...")

# %% [markdown]
# ## Cell 1b — Per-language coverage summary
#
# For every language: N utterances, N speakers, VAD coverage %, and validity %
# for each core acoustic feature (F0, intensity, speech rate). Prints a table
# and dumps to `results.json["coverage"]`. Gives a one-shot view of data health.

# %%
print("\n=== Cell 1b · Per-language coverage summary ===")
_cov_rows = []
_cov_json: dict = {}
_gender_rows: list = []
for _L in ALL_LANGS:
    _fp = idir / f"{_L}_features.tsv"
    if not _fp.exists():
        print(f"[{_L}] SKIP: {_fp.name} not found"); continue
    _df = pd.read_csv(_fp, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None)
    _n = len(_df)
    _n_spk = _df["speaker_id"].nunique() if "speaker_id" in _df else 0
    _row: dict = {"lang": _L, "N_utt": _n, "N_spk": _n_spk}

    # Feature validity (not-null %) — use whichever columns exist
    for _feat, _short in [("f0_raw","F0%"), ("intensity_raw","Int%"),
                          ("speechrate_wps","SR%"), ("hnr_utt","HNR%")]:
        if _feat in _df.columns:
            _pct = 100.0 * _df[_feat].notna().sum() / _n
            _row[_short] = round(_pct, 1)

    # VAD coverage: read {L}_vad.tsv
    _vtsv = idir / f"{_L}_vad.tsv"
    if _vtsv.exists():
        _vdf = pd.read_csv(_vtsv, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None)
        _n_val_utt = int(_vdf["valence"].notna().sum())
        _row["VADutt%"] = round(100.0 * _n_val_utt / len(_vdf), 1) if len(_vdf) else 0.0
        # Word-level: n_covered vs n_words per utterance → mean of ratios
        if "vad_n_covered" in _vdf.columns and "utterance_id" in _vdf.columns:
            _mrg = _vdf[["utterance_id", "vad_n_covered"]].merge(
                _df[["utterance_id", "n_words"]], on="utterance_id", how="inner")
            _mrg = _mrg[_mrg["n_words"] > 0]
            if len(_mrg):
                _mrg["_ratio"] = _mrg["vad_n_covered"] / _mrg["n_words"]
                _row["VADword%"] = round(100.0 * _mrg["_ratio"].mean(), 1)
                _row["MeanCovered"] = round(_mrg["vad_n_covered"].mean(), 1)
                _row["MeanNwords"]  = round(_mrg["n_words"].mean(), 1)

    # Gender split for the summary
    if "gender" in _df.columns:
        _g = _df["gender"].str.lower().str[0].value_counts().to_dict()
        _row["M/F"] = f"{_g.get('m',0):,}/{_g.get('f',0):,}"

    _cov_rows.append(_row)
    _cov_json[_L] = _row

_cov_df = pd.DataFrame(_cov_rows)
print(_cov_df.to_string(index=False))

# Cross-lang averages (unweighted across langs)
if _cov_rows:
    _avg: dict = {"lang": "AVG"}
    for _k in _cov_rows[0]:
        if _k == "lang": continue
        _vals = [r[_k] for r in _cov_rows if isinstance(r.get(_k), (int, float))]
        if _vals: _avg[_k] = round(float(np.mean(_vals)), 1)
    print("--- unweighted average across languages ---")
    print(pd.DataFrame([_avg]).to_string(index=False))
    _cov_json["_AVG"] = _avg

results_json["coverage"] = _cov_json

# %% [markdown]
# ## Cell 2 — Envelope Viewer (Praat NPZ)
#
# Loads `{LANG}_praat_envelopes.npz` and plots N random utterances.
# Each shows: F0 track, Intensity track, per-word F1/F2/F3 medians.
# Word boundaries are drawn as vertical lines.

# %%
for _L in LANGS:
    print(f"\n=== Cell 2 · {_L} · Praat envelope viewer ===")
    npz_path = idir / f"{_L}_praat_envelopes.npz"
    if not npz_path.exists():
        print(f"[{_L}] SKIP: {npz_path.name} not found. Run 20_extract_praat.py first.")
        continue

    _feats_path_L = idir / f"{_L}_features.tsv"
    _df_feats_L = pd.read_csv(_feats_path_L, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None) \
                  if _feats_path_L.exists() else pd.DataFrame()

    data = np.load(npz_path, allow_pickle=True)
    # CRITICAL: cache arrays. NpzFile.__getitem__ decompresses + unpickles the
    # ENTIRE object array on every call. Any `data[key]` inside a loop is a
    # full disk-to-memory hit per iteration.
    _uids_all       = data["utterance_ids"]
    _f0_times       = data["f0_times"]
    _f0_values      = data["f0_values"]
    _int_times      = data["intensity_times"]
    _int_values     = data["intensity_values"]
    _word_starts    = data["word_starts"]
    _word_ends      = data["word_ends"]
    _f1_word_median = data["f1_word_median"]
    _f2_word_median = data["f2_word_median"]
    _f3_word_median = data["f3_word_median"]

    uids = list(_uids_all)
    if TEST_RUN:
        uids = uids[:TEST_RUN_N]
    uid2idx = {uid: i for i, uid in enumerate(uids)}

    # Build a lookup from features TSV
    uid2meta = {}
    if not _df_feats_L.empty:
        for _, row in _df_feats_L.iterrows():
            uid2meta[row["utterance_id"]] = row

    # Sample N utterances that have F0 data
    candidates = [uid for uid in uids if _f0_values[uid2idx[uid]].size > 0]
    sample_uids = rng.sample(candidates, min(N_EXAMPLES, len(candidates)))
    print(f"[{_L}] Sampling {len(sample_uids)} envelope examples from {len(candidates):,} candidates")

    # Load pause tiers for just the sampled uids (streaming, no full load)
    _sampled_set = set(sample_uids)
    uid2pauses: dict = {}
    _filt_jsn_p = idir / f"{_L}_filtered.jsonl"
    if _filt_jsn_p.exists() and _sampled_set:
        with open(_filt_jsn_p, encoding="utf-8") as _pf:
            for _pl in _pf:
                if not _pl.strip(): continue
                _pr = json.loads(_pl)
                if _pr.get("utterance_id") in _sampled_set:
                    uid2pauses[_pr["utterance_id"]] = (
                        _pr.get("silent_pauses") or [],
                        _pr.get("filled_pauses") or [],
                    )
                    if len(uid2pauses) >= len(_sampled_set): break

    for _ex_i, uid in enumerate(sample_uids, 1):
        i = uid2idx[uid]
        f0_t  = _f0_times[i].astype(float)
        f0_v  = _f0_values[i].astype(float)
        int_t = _int_times[i].astype(float)
        int_v = _int_values[i].astype(float)
        w_s   = _word_starts[i].astype(float)
        w_e   = _word_ends[i].astype(float)
        f1_w  = _f1_word_median[i].astype(float)
        f2_w  = _f2_word_median[i].astype(float)
        f3_w  = _f3_word_median[i].astype(float)

        meta = uid2meta.get(uid, {})
        _fmt = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) and v == v else "?"
        sent  = meta.get("sentiment_score")
        label = meta.get("sentiment_label", "?")
        n_w   = meta.get("n_words", "?")
        f0_sc = meta.get("f0_raw")
        spk   = meta.get("speaker_id", "?")
        logprint(f"[{_L}] example {_ex_i}/{len(sample_uids)} uid={uid}",
                 f"spk={spk} sent={_fmt(sent)} label={label} n_words={n_w} f0_raw={_fmt(f0_sc)}",
                 indent=1)

        fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=False)
        fig.suptitle(
            f"[{_L}] {uid}  |  spk={spk}  sent={_fmt(sent)} ({label})  n_words={n_w}  f0_raw={_fmt(f0_sc)}",
            fontsize=9, y=1.01
        )

        _silent, _filled = uid2pauses.get(uid, ([], []))

        def _draw_pauses(ax_):
            for _p in _silent:
                _s, _e = _p.get("time_s", 0), _p.get("time_e", 0)
                ax_.axvspan(_s, _e, color="lightblue", alpha=0.4, zorder=0)
            for _p in _filled:
                _s, _e = _p.get("time_s", 0), _p.get("time_e", 0)
                ax_.axvspan(_s, _e, color="orange", alpha=0.30, zorder=0)

        # F0 envelope
        ax = axes[0]
        _draw_pauses(ax)
        f0_valid = ~np.isnan(f0_v)
        if f0_valid.any():
            ax.plot(f0_t[f0_valid], f0_v[f0_valid], color="#E63946", lw=1.2, label="F0")
        for s, e in zip(w_s, w_e):
            ax.axvline(s, color="gray", lw=0.5, alpha=0.4)
        ax.set_ylabel("F0 (Hz)")
        ax.set_title(f"F0 envelope  ·  silent pauses (blue) + filled pauses (orange) shaded  "
                     f"·  n_silent={len(_silent)} n_filled={len(_filled)}", fontsize=9)

        # Intensity envelope
        ax = axes[1]
        _draw_pauses(ax)
        int_valid = ~np.isnan(int_v)
        if int_valid.any():
            ax.plot(int_t[int_valid], int_v[int_valid], color="#457B9D", lw=1.2, label="Intensity")
        for s, e in zip(w_s, w_e):
            ax.axvline(s, color="gray", lw=0.5, alpha=0.4)
        ax.set_ylabel("Intensity (dB)")
        ax.set_title("Intensity envelope")

        # Per-word F1/F2/F3 medians (step plot over word spans)
        ax = axes[2]
        for k, (vals, color, lbl) in enumerate(zip(
            [f1_w, f2_w, f3_w],
            ["#E63946", "#2A9D8F", "#E9C46A"],
            ["F1", "F2", "F3"],
        )):
            for j, (s, e, v) in enumerate(zip(w_s, w_e, vals)):
                if not np.isnan(v):
                    ax.hlines(v, s, e, colors=color, lw=2.5, label=lbl if j == 0 else "")
        ax.set_ylabel("Formant (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title("Per-word F1/F2/F3 medians")
        ax.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        save_fig(f"cell02_envelope_{_L}_example{_ex_i:02d}_{uid[:60]}")

# %% [markdown]
# ## Cell 2b — VAD Word Envelope Viewer
#
# Loads `{LANG}_vad_envelopes.npz` and plots per-word valence/arousal/dominance
# for N random utterances. NaN = word not in lexicon (non-content UPOS or OOV).
# Aligned to word timing from v4 words[], same x-axis as F0 envelopes.

# %%
for _L in LANGS:
    print(f"\n=== Cell 2b · {_L} · VAD envelope viewer ===")
    vad_npz_path = idir / f"{_L}_vad_envelopes.npz"
    vad_tsv_path = idir / f"{_L}_vad.tsv"
    if not vad_npz_path.exists():
        print(f"[{_L}] SKIP: {vad_npz_path.name} not found. Run 35_vad.py with save_vad_envelopes=true.")
        continue

    vdata = np.load(vad_npz_path, allow_pickle=True)
    # Cache arrays (see Cell 2 comment — NpzFile decompresses per __getitem__)
    _v_uids_all  = vdata["utterance_ids"]
    _v_wstarts   = vdata["word_starts"]
    _v_wends     = vdata["word_ends"]
    _v_valences  = vdata["word_valences"]
    _v_arousals  = vdata["word_arousals"]
    _v_dominance = vdata["word_dominances"]

    vad_uids = list(_v_uids_all)
    if TEST_RUN:
        vad_uids = vad_uids[:TEST_RUN_N]
    vuid2idx = {uid: i for i, uid in enumerate(vad_uids)}

    # Load utterance-level VAD scores for metadata
    df_vad = pd.read_csv(vad_tsv_path, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None) if vad_tsv_path.exists() else pd.DataFrame()
    uid2vad = {}
    if not df_vad.empty:
        for _, row in df_vad.iterrows():
            uid2vad[row["utterance_id"]] = row

    # Rebuild uid2meta from this language's features TSV
    _feats_path_L = idir / f"{_L}_features.tsv"
    _df_feats_L = pd.read_csv(_feats_path_L, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None) \
                  if _feats_path_L.exists() else pd.DataFrame()
    uid2meta = {}
    if not _df_feats_L.empty:
        for _, row in _df_feats_L.iterrows():
            uid2meta[row["utterance_id"]] = row

    # Sample utterances with ≥1 covered word
    candidates_v = [
        uid for uid in vad_uids
        if np.any(~np.isnan(_v_valences[vuid2idx[uid]].astype(float)))
    ]
    sample_v_uids = rng.sample(candidates_v, min(N_EXAMPLES, len(candidates_v)))
    print(f"[{_L}] Sampling {len(sample_v_uids)} VAD examples from {len(candidates_v):,} covered utterances")

    for _ex_i, uid in enumerate(sample_v_uids, 1):
        i = vuid2idx[uid]
        starts    = _v_wstarts[i].astype(float)
        ends      = _v_wends[i].astype(float)
        valences  = _v_valences[i].astype(float)
        arousals  = _v_arousals[i].astype(float)
        dominances = _v_dominance[i].astype(float)

        meta = uid2meta.get(uid, {})
        vad_meta = uid2vad.get(uid, {})
        _fmt = lambda v, p=2: f"{v:.{p}f}" if isinstance(v, (int, float)) and v == v else "?"
        sent  = meta.get("sentiment_score")
        utt_v = vad_meta.get("valence")
        utt_a = vad_meta.get("arousal")
        n_cov = int(vad_meta.get("vad_n_covered", 0))
        n_w   = len(starts)
        pct_cov = 100.0 * n_cov / n_w if n_w else 0.0
        logprint(f"[{_L}] VAD example {_ex_i}/{len(sample_v_uids)} uid={uid}",
                 f"sent={_fmt(sent)} utt_v={_fmt(utt_v,3)} utt_a={_fmt(utt_a,3)} "
                 f"covered={n_cov}/{n_w} ({pct_cov:.0f}%)",
                 indent=1)

        fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
        fig.suptitle(
            f"[{_L}] {uid}  |  sent={_fmt(sent)}  utt_valence={_fmt(utt_v,3)}  "
            f"utt_arousal={_fmt(utt_a,3)}  covered={n_cov}/{n_w} words ({pct_cov:.0f}%)",
            fontsize=9, y=1.01
        )

        for ax, vals, color, label, ylim in zip(
            axes,
            [valences, arousals, dominances],
            ["#E63946", "#2A9D8F", "#E9C46A"],
            ["Valence", "Arousal", "Dominance"],
            [(0, 1), (0, 1), (0, 1)],
        ):
            # Step plot: horizontal bar per word, NaN = white gap
            for j, (s, e, v) in enumerate(zip(starts, ends, vals)):
                if not np.isnan(s) and not np.isnan(e) and not np.isnan(v):
                    ax.hlines(v, s, e, colors=color, lw=3.0, alpha=0.85)
                    ax.vlines(s, 0, v, colors=color, lw=0.5, alpha=0.3)
            # Mean line
            valid_v = vals[~np.isnan(vals)]
            if len(valid_v) > 0:
                ax.axhline(float(np.mean(valid_v)), color=color, lw=1.0,
                           ls="--", alpha=0.6, label=f"mean={np.mean(valid_v):.3f}")
            ax.set_ylabel(label, fontsize=9)
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color="gray", lw=0.5, alpha=0.4)
            ax.legend(fontsize=8, loc="upper right")

        axes[-1].set_xlabel("Time (s)", fontsize=9)
        plt.tight_layout()
        save_fig(f"cell02b_vadenv_{_L}_example{_ex_i:02d}_{uid[:60]}")


# %% [markdown]
# ## Cell 3 — Speechrate + Transcript Viewer
#
# For each sampled utterance: word timeline bars (colored by duration),
# silent_pauses as shaded regions, transcript text with vowels marked.

# %%
# Vowel sets for transcript highlighting only (synced with extraction.py)
VOWELS = {
    "HR": set("aeiouAEIOUáéíóúÁÉÍÓÚàèìòùÀÈÌÒÙ"),
    "CZ": set("aeiouAEIOUáéíóúýÁÉÍÓÚÝůŮěĚ"),
    "PL": set("aeiouAEIOUáéíóúÁÉÍÓÚąęóĄĘÓ"),
    "RS": set("aeiouAEIOUáéíóúÁÉÍÓÚàèìòùÀÈÌÒÙ"),
    "SI": set("aeiouAEIOUáéíóúÁÉÍÓÚ"),
}

for _L in LANGS:
    print(f"\n=== Cell 3 · {_L} · Speechrate + transcript viewer ===")
    jsonl_path = idir / f"{_L}_filtered.jsonl"
    if not jsonl_path.exists():
        print(f"[{_L}] SKIP: {jsonl_path.name} not found.")
        continue
    lang_vowels = VOWELS.get(_L, set("aeiouAEIOU"))

    _feats_path_L = idir / f"{_L}_features.tsv"
    _df_feats_L = pd.read_csv(_feats_path_L, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None) \
                  if _feats_path_L.exists() else pd.DataFrame()
    _uid2meta_L = {r["utterance_id"]: r for _, r in _df_feats_L.iterrows()} if not _df_feats_L.empty else {}

    if TEST_RUN:
        records = []
        with open(jsonl_path, encoding="utf-8") as _f:
            for _line in _f:
                if len(records) >= TEST_RUN_N:
                    break
                _line = _line.strip()
                if _line:
                    records.append(json.loads(_line))
        print(f"[{_L}] TEST_RUN cap: loaded {len(records):,} records (max {TEST_RUN_N:,}).")
    else:
        records = load_jsonl(jsonl_path)
        print(f"[{_L}] Loaded {len(records):,} filtered records.")

    # Filter to utterances that have speechrate in features and word timing
    has_words = [r for r in records if r.get("words_align")]
    if not _df_feats_L.empty and "speechrate_wps" in _df_feats_L.columns:
        sr_uids = set(_df_feats_L.dropna(subset=["speechrate_wps"])["utterance_id"])
        has_words = [r for r in has_words if r["utterance_id"] in sr_uids]

    sample_recs = rng.sample(has_words, min(N_EXAMPLES, len(has_words)))
    print(f"[{_L}] Sampling {len(sample_recs)} transcript examples from {len(has_words):,} candidates")

    for _ex_i, rec in enumerate(sample_recs, 1):
        uid   = rec["utterance_id"]
        text  = rec.get("text", "")
        words = rec.get("words_align", [])
        pauses = rec.get("silent_pauses") or []
        meta  = _uid2meta_L.get(uid, {}) if _uid2meta_L else {}
        _fmt = lambda v, p=2: f"{v:.{p}f}" if isinstance(v, (int, float)) and v == v else "?"
        sr    = meta.get("speechrate_wps")
        sent  = meta.get("sentiment_score")
        n_paus = len(pauses)

        if not words:
            continue

        # Timeline plot
        fig, ax = plt.subplots(figsize=(14, 2.5))
        durations = [w.get("end", 0) - w.get("start", 0) for w in words]
        max_dur = max(durations) if durations else 1
        # Color = per-word duration. Light yellow = short (fast articulation),
        # dark red = long (slow articulation / lengthened word).
        cmap = plt.get_cmap("YlOrRd")

        for w in words:
            s = w.get("start", 0)
            e = w.get("end", 0)
            word_text = w.get("word", "")
            dur = e - s
            color = cmap(min(dur / max_dur, 1.0))
            ax.barh(0, dur, left=s, height=0.6, color=color, edgecolor="white", linewidth=0.3)
            ax.text((s + e) / 2, 0, word_text, ha="center", va="center",
                    fontsize=7, color="black", clip_on=True)

        # Silent pauses (light blue) and filled pauses (light orange) as shaded spans
        for p in pauses:
            ps = p.get("time_s", p.get("start", 0))
            pe = p.get("time_e", p.get("end", ps))
            ax.axvspan(ps, pe, color="lightblue", alpha=0.5)
        _filled = rec.get("filled_pauses") or []
        for p in _filled:
            ps = p.get("time_s", p.get("start", 0))
            pe = p.get("time_e", p.get("end", ps))
            ax.axvspan(ps, pe, color="orange", alpha=0.35)

        ax.set_yticks([])
        ax.set_xlabel(f"Time (s) — word color: light=short duration (fast), dark=long (slow)  ·  "
                      f"blue span=silent pause  ·  orange span=filled pause",
                      fontsize=7)
        ax.set_title(
            f"[{_L}] {uid}  |  sent={_fmt(sent)}  sr={_fmt(sr)} wps  "
            f"n_silent={n_paus}  n_filled={len(_filled)}",
            fontsize=9,
        )
        plt.tight_layout()
        save_fig(f"cell03_timeline_{_L}_example{_ex_i:02d}_{uid[:60]}")

        # Transcript with vowels marked (uppercase vowels)
        marked = ""
        for ch in text:
            marked += ch.upper() if ch in lang_vowels else ch
        print(f"  {marked[:200]}")
        print()

# %% [markdown]
# ## Cell 4 — Feature Trend Replotters
#
# Select languages and features, plot individual panels and overlay view.

# %%
# ── Configure here ───────────────────────────────
PLOT_LANGS = ["HR", "CZ", "PL", "RS", "SI"]
PLOT_FEATS = ["f0_raw", "speechrate_wps", "intensity_norm"]
N_BINS     = 60
# ─────────────────────────────────────────────────

FEAT_LABELS = {
    "f0_raw": "F0 (Hz)", "f0_norm": "F0 (norm.)",
    "intensity_raw": "Intensity (dB)", "intensity_norm": "Intensity (dB, norm.)",
    "speechrate_wps": "Speech rate (wps)", "speechrate_sps": "Speech rate (sps)",
}

binned_all: dict = {}
for lang in PLOT_LANGS:
    fp = idir / f"{lang}_features.tsv"
    if not fp.exists():
        print(f"[{lang}] features TSV not found")
        continue
    df = pd.read_csv(fp, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None)
    df["bin"] = pd.cut(df["sentiment_score"],
                       bins=np.linspace(0, 5, N_BINS + 1),
                       labels=False, include_lowest=True)
    for feat in PLOT_FEATS:
        if feat not in df.columns:
            continue
        means = df.dropna(subset=[feat]).groupby("bin")[feat].mean().reindex(range(N_BINS))
        binned_all[(lang, feat)] = means

x_vals = np.linspace(0, 5, N_BINS)

# (A) Individual panels: features × languages
n_f, n_l = len(PLOT_FEATS), len(PLOT_LANGS)
fig, axes = plt.subplots(n_f, n_l, figsize=(n_l * 3.5, n_f * 3), squeeze=False)
fig.suptitle("Individual Trend Panels", fontsize=12)
for r, feat in enumerate(PLOT_FEATS):
    for c, lang in enumerate(PLOT_LANGS):
        ax = axes[r][c]
        key = (lang, feat)
        curve = binned_all.get(key)
        if curve is not None and not curve.isna().all():
            y = curve.values
            ax.plot(x_vals, y, color=PALETTE.get(lang, "#888"), lw=1.8)
            valid = ~np.isnan(y)
            if valid.sum() > 2:
                sl, ic, *_ = stats.linregress(x_vals[valid], y[valid])
                ax.plot(x_vals[valid], sl * x_vals[valid] + ic,
                        color="#457B9D", lw=1.0, ls="--")
        if r == 0: ax.set_title(lang, fontsize=10)
        if c == 0: ax.set_ylabel(FEAT_LABELS.get(feat, feat), fontsize=8)
        if r == n_f - 1: ax.set_xlabel("Sentiment", fontsize=8)
        ax.tick_params(labelsize=7)
plt.tight_layout()
save_fig("cell04_trend_individual_panels")

# (B) Overlay: one subplot per feature, all languages colored
fig, axes = plt.subplots(1, n_f, figsize=(n_f * 5, 4))
if n_f == 1: axes = [axes]
fig.suptitle("Language Overlay (min-max normalised per language)", fontsize=12)
for ax, feat in zip(axes, PLOT_FEATS):
    for lang in PLOT_LANGS:
        curve = binned_all.get((lang, feat))
        if curve is None or curve.isna().all():
            continue
        y = curve.values.astype(float)
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        if ymax > ymin:
            y = (y - ymin) / (ymax - ymin)
        ax.plot(x_vals, y, color=PALETTE.get(lang, "#888"), lw=1.8, label=lang, alpha=0.9)
    ax.set_title(FEAT_LABELS.get(feat, feat), fontsize=10)
    ax.set_xlabel("Sentiment", fontsize=9)
    ax.set_ylabel("Normalised", fontsize=9)
    ax.legend(fontsize=8, loc="best")
plt.tight_layout()
save_fig("cell04_trend_language_overlay")

# %% [markdown]
# ## Cell 5 — Praat vs OpenSMILE Comparison
#
# Scatter-compares equivalent features from both extractors.
# Uses LLD NPZ if available (computes mean voiced F0), else scalar functionals TSV.

# %%
for _L in LANGS:
  print(f"\n=== Cell 5 · {_L} · Praat vs OpenSMILE (LLD-mean) comparison ===")
  osm_path = idir / f"{_L}_opensmile.tsv"
  lld_path = idir / f"{_L}_opensmile_lld.npz"
  _feats_path_L = idir / f"{_L}_features.tsv"
  _df_feats_L = pd.read_csv(_feats_path_L, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None) \
                if _feats_path_L.exists() else pd.DataFrame()

  if not osm_path.exists() and not lld_path.exists():
    print(f"[{_L}] SKIP: No OpenSMILE outputs found.")
    continue
  if _df_feats_L.empty:
    print(f"[{_L}] SKIP: Features TSV not available.")
    continue

  if lld_path.exists():
    lld = np.load(lld_path, allow_pickle=True)
    _lld_uids_all = lld["utterance_ids"]
    _f0_lld   = lld["f0_lld"]
    _loud_lld = lld["loudness_lld"]
    _f1_lld   = lld["f1_lld"]
    _f2_lld   = lld["f2_lld"]
    _f3_lld   = lld["f3_lld"]

    lld_uids = list(_lld_uids_all)
    if TEST_RUN:
        lld_uids = lld_uids[:TEST_RUN_N]

    def mean_voiced(arr):
        v = arr.astype(float)
        v[v <= 0] = np.nan
        return np.nanmean(v) if np.any(~np.isnan(v)) else np.nan

    f0_osm, loud_osm, f1_osm, f2_osm, f3_osm, uids_osm = [], [], [], [], [], []
    for uid, f0_arr, loud_arr, f1_arr, f2_arr, f3_arr in zip(
        lld_uids, _f0_lld, _loud_lld, _f1_lld, _f2_lld, _f3_lld,
    ):
        f0_osm.append(mean_voiced(f0_arr))
        loud_osm.append(float(np.nanmean(loud_arr.astype(float))))
        f1_osm.append(mean_voiced(f1_arr))
        f2_osm.append(mean_voiced(f2_arr))
        f3_osm.append(mean_voiced(f3_arr))
        uids_osm.append(uid)

    df_osm = pd.DataFrame({
        "utterance_id": uids_osm,
        "osm_f0": f0_osm, "osm_loudness": loud_osm,
        "osm_f1": f1_osm, "osm_f2": f2_osm, "osm_f3": f3_osm,
    })
    df_merged = _df_feats_L.merge(df_osm, on="utterance_id", how="inner")

    pairs = [
        ("f0_raw",        "osm_f0",      "F0 raw vs OSM F0 (LLD mean voiced, log-semitones)"),
        ("intensity_norm","osm_loudness", "Intensity norm vs OSM Loudness (LLD mean)"),
        ("f1_median",     "osm_f1",       "F1 median vs OSM F1 (LLD mean voiced)"),
        ("f2_median",     "osm_f2",       "F2 median vs OSM F2 (LLD mean voiced)"),
        ("f3_median",     "osm_f3",       "F3 median vs OSM F3 (LLD mean voiced)"),
    ]
  else:
    df_osm = pd.read_csv(osm_path, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None)
    df_merged = _df_feats_L.merge(df_osm, on="utterance_id", how="inner")
    pairs = [
        ("f0_raw", "osmile_F0semitoneFrom27.5Hz_sma3nz_amean",
         "F0 raw vs OSM F0 (functional mean)"),
        ("intensity_norm", "osmile_Loudness_sma3_amean",
         "Intensity norm vs OSM Loudness (functional mean)"),
    ]

  valid_pairs = [(a, b, lbl) for a, b, lbl in pairs
                 if a in df_merged.columns and b in df_merged.columns]
  if not valid_pairs:
    print(f"[{_L}] No matching column pairs found for comparison.")
    continue

  fig, axes = plt.subplots(1, len(valid_pairs), figsize=(len(valid_pairs) * 4, 4))
  if len(valid_pairs) == 1: axes = [axes]
  fig.suptitle(f"{_L} — Praat vs OpenSMILE Feature Comparison", fontsize=11)

  for ax, (praat_col, osm_col, label) in zip(axes, valid_pairs):
    sub = df_merged[[praat_col, osm_col]].dropna()
    if len(sub) < 10:
        ax.set_title(f"{label}\n(n<10)"); continue
    r, p = stats.spearmanr(sub[praat_col], sub[osm_col])
    ax.scatter(sub[praat_col], sub[osm_col],
               alpha=0.15, s=8, color=PALETTE.get(_L, "#888"))
    ax.set_xlabel(f"Praat: {praat_col}", fontsize=8)
    ax.set_ylabel(f"OSM: {osm_col.split('_')[-1]}", fontsize=8)
    ax.set_title(f"{label}\nSpearman r={r:.3f}, p={p:.3e}", fontsize=8)
    ax.tick_params(labelsize=7)
    logprint(f"  [{_L}] {label}", f"ρ={r:+.4f} p={p:.3e} n={len(sub):,}")

  plt.tight_layout()
  save_fig(f"cell05_praat_vs_opensmile_{_L}")

# %% [markdown]
# ## Cell 6 — Per-Language Investigator / SI Anomaly Check
#
# All languages overlaid per feature. Marks where SI deviates most.
# Also prints a summary table.

# %%
ALL_LANGS = ["HR", "CZ", "PL", "RS", "SI"]
INV_FEATS = ["f0_raw", "speechrate_wps", "intensity_norm"]

binned_inv: dict = {}
df_all: dict = {}
for lang in ALL_LANGS:
    fp = idir / f"{lang}_features.tsv"
    if not fp.exists():
        continue
    df = pd.read_csv(fp, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None)
    df["bin"] = pd.cut(df["sentiment_score"],
                       bins=np.linspace(0, 5, N_BINS + 1),
                       labels=False, include_lowest=True)
    df_all[lang] = df
    for feat in INV_FEATS:
        if feat not in df.columns:
            continue
        means = df.dropna(subset=[feat]).groupby("bin")[feat].mean().reindex(range(N_BINS))
        binned_inv[(lang, feat)] = means

# Summary table
print("Language summary:")
print(f"{'Lang':6s} {'N_utt':>8s} {'N_spk':>6s} {'MeanSent':>10s} {'MeanF0':>8s} {'MeanSR':>8s}")
for lang, df in df_all.items():
    n_utt = len(df)
    n_spk = df["speaker_id"].nunique() if "speaker_id" in df else "?"
    ms = df["sentiment_score"].mean() if "sentiment_score" in df else float("nan")
    mf = df["f0_raw"].mean() if "f0_raw" in df else float("nan")
    mr = df["speechrate_wps"].mean() if "speechrate_wps" in df else float("nan")
    print(f"{lang:6s} {n_utt:>8,} {str(n_spk):>6s} {ms:>10.3f} {mf:>8.1f} {mr:>8.3f}")
print()

# Per-feature overlay + SI deviation
fig, axes = plt.subplots(1, len(INV_FEATS), figsize=(len(INV_FEATS) * 5, 4))
if len(INV_FEATS) == 1: axes = [axes]
fig.suptitle("Per-Language Overlay — SI Anomaly Check", fontsize=12)

for ax, feat in zip(axes, INV_FEATS):
    si_curve = binned_inv.get(("SI", feat))
    other_curves = [binned_inv[(l, feat)] for l in ALL_LANGS
                    if l != "SI" and (l, feat) in binned_inv]

    for lang in ALL_LANGS:
        curve = binned_inv.get((lang, feat))
        if curve is None or curve.isna().all():
            continue
        y = curve.values.astype(float)
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        if ymax > ymin: y = (y - ymin) / (ymax - ymin)
        lw = 2.5 if lang == "SI" else 1.4
        ax.plot(x_vals, y, color=PALETTE.get(lang, "#888"), lw=lw, label=lang,
                alpha=1.0 if lang == "SI" else 0.65)

    # Mark max deviation of SI from others
    if si_curve is not None and other_curves:
        si_y = si_curve.values.astype(float)
        si_norm = (si_y - np.nanmin(si_y)) / (np.nanmax(si_y) - np.nanmin(si_y) + 1e-9)
        oth_norm = np.array([
            (c.values.astype(float) - np.nanmin(c.values)) /
            (np.nanmax(c.values) - np.nanmin(c.values) + 1e-9)
            for c in other_curves
        ])
        mean_oth = np.nanmean(oth_norm, axis=0)
        dev = np.abs(si_norm - mean_oth)
        max_bin = np.nanargmax(dev)
        ax.axvline(x_vals[max_bin], color="#9B5DE5", lw=1.5, ls=":",
                   alpha=0.8, label=f"SI max dev @ sent={x_vals[max_bin]:.2f}")

    ax.set_title(FEAT_LABELS.get(feat, feat), fontsize=10)
    ax.set_xlabel("Sentiment", fontsize=9)
    if ax is axes[0]: ax.set_ylabel("Normalised", fontsize=9)
    ax.legend(fontsize=8, loc="best")

plt.tight_layout()
save_fig("cell06_si_anomaly_overlay")

# %% [markdown]
# ## Cell 6b — Global Trend: Per-language + Equal-Language vs Weighted-Speaker
#
# For each feature: 5 per-language mean curves (thin), plus two "global" curves:
#   * Equal-language (bold black solid): mean of per-language curves.
#     Each language contributes equally regardless of speaker count.
#   * Weighted-speaker (bold gray dashed): pooled speaker-mean-per-bin across langs.
#     Bigger corpora dominate; matches config `global_trend_weighting=weighted_speaker`.
# Reuses `df_all` and `binned_inv` from Cell 6.

# %%
# CRITICAL for F0: males and females have very different absolute pitch ranges.
# Pooling raw F0 across genders inflates within-language variance and can flip
# apparent trends. Toggle SPLIT_BY_GENDER=True for a per-gender view.
# For publication F0 should either be per-gender or per-speaker z-scored before averaging.
SPLIT_BY_GENDER = False   # True → separate M/F curves per language + per-gender global means

if not df_all:
    print("[SKIP] No language DFs loaded (Cell 6 didn't populate df_all).")
elif SPLIT_BY_GENDER:
    _GT_FEATS = INV_FEATS
    _x_gt = 0.5 * (np.linspace(0, 5, N_BINS + 1)[:-1] + np.linspace(0, 5, N_BINS + 1)[1:])

    fig, axes = plt.subplots(2, len(_GT_FEATS), figsize=(len(_GT_FEATS) * 5, 7), squeeze=False)
    fig.suptitle("Global trend by gender — equal-language (solid) vs weighted-speaker (dashed)",
                 fontsize=12)

    for row_i, _g in enumerate(["m", "f"]):
        for col_i, feat in enumerate(_GT_FEATS):
            ax = axes[row_i][col_i]
            _per_lang: dict = {}
            for _lg, _df in df_all.items():
                if feat not in _df.columns or "gender" not in _df.columns: continue
                _sub = _df[_df["gender"].str.lower().str[0] == _g].dropna(subset=[feat, "bin"])
                if _sub.empty: continue
                _c = _sub.groupby("bin")[feat].mean().reindex(range(N_BINS)).values.astype(float)
                _per_lang[_lg] = _c
                ax.plot(_x_gt, _c, color=PALETTE.get(_lg, "#888"), lw=1.1, alpha=0.55, label=_lg)
            if _per_lang:
                _eq = np.nanmean(np.vstack(list(_per_lang.values())), axis=0)
                ax.plot(_x_gt, _eq, color="black", lw=2.2, label="equal-lang")
                # weighted-speaker per gender
                _pooled = []
                for _lg, _df in df_all.items():
                    if feat not in _df.columns or "gender" not in _df.columns: continue
                    _sub = _df[_df["gender"].str.lower().str[0] == _g].dropna(subset=[feat, "bin"])
                    if _sub.empty: continue
                    _pooled.append(_sub.groupby(["speaker_id", "bin"])[feat].mean().reset_index())
                if _pooled:
                    _we = pd.concat(_pooled, ignore_index=True).groupby("bin")[feat].mean() \
                                                                 .reindex(range(N_BINS)).values
                    ax.plot(_x_gt, _we, color="#444", lw=1.8, ls="--", label="weighted-spk")
            _n = sum(((_df["gender"].str.lower().str[0] == _g).sum() if "gender" in _df else 0)
                     for _df in df_all.values())
            ax.set_title(f"{FEAT_LABELS.get(feat, feat)}  ·  {_g.upper()} (n={_n:,})", fontsize=9)
            ax.set_xlabel("Sentiment", fontsize=8)
            if col_i == 0: ax.set_ylabel(f"{_g.upper()}: {feat}", fontsize=8)
            ax.legend(fontsize=6, loc="best", ncol=2)
    plt.tight_layout()
    save_fig("cell06b_global_trend_by_gender")
else:
    _GT_FEATS = INV_FEATS  # same feature set as the SI-anomaly cell
    _x_gt = 0.5 * (np.linspace(0, 5, N_BINS + 1)[:-1] + np.linspace(0, 5, N_BINS + 1)[1:])

    fig, axes = plt.subplots(1, len(_GT_FEATS), figsize=(len(_GT_FEATS) * 5, 4), squeeze=False)
    fig.suptitle("Global trend — equal-language vs weighted-speaker  "
                 "(F0 mixes M+F; toggle SPLIT_BY_GENDER=True for per-gender view)", fontsize=11)
    axes = axes[0]

    for ax, feat in zip(axes, _GT_FEATS):
        # Per-language curves (raw scale, no normalisation — we want to see the range)
        _per_lang_curves: dict = {}
        for _lg in ALL_LANGS:
            _c = binned_inv.get((_lg, feat))
            if _c is None or _c.isna().all():
                continue
            _per_lang_curves[_lg] = _c.values.astype(float)
            ax.plot(_x_gt, _c.values, color=PALETTE.get(_lg, "#888"),
                    lw=1.2, alpha=0.55, label=_lg)

        if not _per_lang_curves:
            ax.set_title(f"{FEAT_LABELS.get(feat, feat)}\n(no data)"); continue

        # Equal-language: mean of per-lang curves (nan-safe)
        _stack = np.vstack(list(_per_lang_curves.values()))
        _equal = np.nanmean(_stack, axis=0)
        ax.plot(_x_gt, _equal, color="black", lw=2.4, label="equal-language")

        # Weighted-speaker: per-speaker-bin means pooled across all langs, then mean per bin.
        # Approximates paper's weighted_speaker option.
        _pooled = []
        for _lg, _df in df_all.items():
            if feat not in _df.columns or "speaker_id" not in _df.columns: continue
            _sub = _df.dropna(subset=[feat, "bin"])
            _spk_bin = _sub.groupby(["speaker_id", "bin"])[feat].mean().reset_index()
            _pooled.append(_spk_bin)
        if _pooled:
            _all_spk = pd.concat(_pooled, ignore_index=True)
            _weighted = _all_spk.groupby("bin")[feat].mean().reindex(range(N_BINS)).values
            ax.plot(_x_gt, _weighted, color="#444", lw=2.0, ls="--", label="weighted-speaker")

        ax.set_title(FEAT_LABELS.get(feat, feat), fontsize=10)
        ax.set_xlabel("Sentiment", fontsize=9)
        ax.set_ylabel(FEAT_LABELS.get(feat, feat), fontsize=9)
        ax.legend(fontsize=7, loc="best", ncol=2)

    plt.tight_layout()
    save_fig("cell06b_global_trend_pooled")

    # Quick numeric summary — divergence between the two weightings
    print("\nEqual-language vs weighted-speaker divergence (mean |Δ| across bins):")
    print(f"{'Feature':<20s} {'mean |Δ|':>10s} {'max |Δ|':>10s} {'range(equal)':>14s}")
    for feat in _GT_FEATS:
        _cs = [binned_inv[(l, feat)].values.astype(float) for l in ALL_LANGS
               if (l, feat) in binned_inv]
        if len(_cs) < 2: continue
        _eq = np.nanmean(np.vstack(_cs), axis=0)
        _pooled = []
        for _lg, _df in df_all.items():
            if feat not in _df.columns: continue
            _sub = _df.dropna(subset=[feat, "bin"])
            _pooled.append(_sub.groupby(["speaker_id", "bin"])[feat].mean().reset_index())
        if not _pooled: continue
        _we = pd.concat(_pooled, ignore_index=True).groupby("bin")[feat].mean() \
                                                    .reindex(range(N_BINS)).values
        _d = np.abs(_eq - _we)
        print(f"{FEAT_LABELS.get(feat, feat):<20s} {np.nanmean(_d):>10.4f} "
              f"{np.nanmax(_d):>10.4f} {np.nanmax(_eq) - np.nanmin(_eq):>14.4f}")

# %% [markdown]
# ## Cell 6c — Clean editorial trend plots (per-lang and global)
#
# Editorial style à la `3i_editorial_plots_png-GOOD.py`: single thick orange line,
# no per-language overlays, no dashed alternates. One multi-panel figure per view.
#
# Views produced (18 figures total when ALL_LANGS = 5):
#
#   PER-LANG (5 langs × 3 variants = 15):
#     · <lang>_pooled_zscore  — F0 z-scored per speaker (gender-neutral), other
#       features as raw. All speakers pooled.
#     · <lang>_gender_F       — F speakers only, F0 raw and other features raw.
#     · <lang>_gender_M       — M speakers only.
#
#   GLOBAL (all languages pooled, 3 variants):
#     · GLOBAL_pooled_zscore, GLOBAL_gender_F, GLOBAL_gender_M
#
# Records slopes / intercepts / r into results.json["editorial_trends"].

# %%
_EDIT_FEATS = INV_FEATS  # ["f0_raw", "speechrate_wps", "intensity_norm"]
_x_edit = 0.5 * (np.linspace(0, 5, N_BINS + 1)[:-1] + np.linspace(0, 5, N_BINS + 1)[1:])
results_json.setdefault("editorial_trends", {})

def _plot_editorial_panel(df, feats, title_prefix, save_stub, feat_transforms=None):
    """One figure, N panels (one per feature); orange thick line; log stats.
    feat_transforms: {feat: series_transform_func} — e.g. z-score F0.
    """
    feat_transforms = feat_transforms or {}
    fig, axes = plt.subplots(1, len(feats), figsize=(len(feats) * 5, 4), squeeze=False)
    axes = axes[0]
    _ent = {"n_utterances": int(len(df)),
            "n_speakers": int(df["speaker_id"].nunique()) if "speaker_id" in df else None,
            "features": {}}
    for _ax, _feat in zip(axes, feats):
        if _feat not in df.columns:
            _ax.set_title(f"{FEAT_LABELS.get(_feat, _feat)}\n(missing)"); continue
        _work = df.copy()
        _label_extra = ""
        if _feat in feat_transforms:
            _work[_feat] = feat_transforms[_feat](_work)
            _label_extra = " (per-speaker z)"
        _y = binned_speaker_mean(_work, _feat, N_BINS)
        _stats = clean_trend_plot(
            _ax, _x_edit, _y,
            title=f"{FEAT_LABELS.get(_feat, _feat)}{_label_extra}",
            ylabel=f"{_feat}{_label_extra}",
        )
        _ent["features"][_feat] = {**_stats}
        logprint(f"  [{title_prefix}] {_feat}",
                 f"slope={_stats['slope']:+.4f} r={_stats['r']:+.4f} n_bins_ok={_stats['n_bins_valid']}",
                 indent=1)
    fig.suptitle(title_prefix, fontsize=12)
    plt.tight_layout()
    save_fig(save_stub)
    return _ent

# Per-language passes
for _L in ALL_LANGS:
    _fp = idir / f"{_L}_features.tsv"
    if not _fp.exists():
        continue
    _df_L = pd.read_csv(_fp, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None)
    print(f"\n=== Cell 6c · {_L} · editorial trends ===")

    # Variant 1: pooled with F0 z-scored per speaker
    _ent = _plot_editorial_panel(
        _df_L, _EDIT_FEATS,
        title_prefix=f"{_L} — pooled  ·  F0 per-speaker z-scored",
        save_stub=f"cell06c_{_L}_pooled_zscore",
        feat_transforms={"f0_raw": lambda d: zscore_per_speaker(d, "f0_raw")},
    )
    results_json["editorial_trends"].setdefault(_L, {})["pooled_zscore"] = _ent

    # Variants 2 & 3: gender-split, raw F0
    if "gender" in _df_L.columns:
        for _g, _tag in [("f", "F"), ("m", "M")]:
            _sub = _df_L[_df_L["gender"].str.lower().str[0] == _g]
            if _sub.empty:
                print(f"[{_L}] no {_tag} speakers; skipping"); continue
            _ent = _plot_editorial_panel(
                _sub, _EDIT_FEATS,
                title_prefix=f"{_L} — {_tag} speakers only  ·  F0 raw",
                save_stub=f"cell06c_{_L}_gender_{_tag}",
            )
            results_json["editorial_trends"][_L][f"gender_{_tag}"] = _ent

# GLOBAL: pool all languages together
_globals_ok = [pd.read_csv(idir / f"{_L}_features.tsv", sep="\t",
                             nrows=TEST_RUN_N if TEST_RUN else None)
                for _L in ALL_LANGS if (idir / f"{_L}_features.tsv").exists()]
if _globals_ok:
    _df_g = pd.concat(_globals_ok, ignore_index=True)
    print(f"\n=== Cell 6c · GLOBAL (all {len(_globals_ok)} langs merged) · editorial trends ===")
    _ent = _plot_editorial_panel(
        _df_g, _EDIT_FEATS,
        title_prefix="GLOBAL — pooled  ·  F0 per-speaker z-scored",
        save_stub="cell06c_GLOBAL_pooled_zscore",
        feat_transforms={"f0_raw": lambda d: zscore_per_speaker(d, "f0_raw")},
    )
    results_json["editorial_trends"].setdefault("GLOBAL", {})["pooled_zscore"] = _ent
    if "gender" in _df_g.columns:
        for _g, _tag in [("f", "F"), ("m", "M")]:
            _sub = _df_g[_df_g["gender"].str.lower().str[0] == _g]
            if _sub.empty: continue
            _ent = _plot_editorial_panel(
                _sub, _EDIT_FEATS,
                title_prefix=f"GLOBAL — {_tag} speakers only  ·  F0 raw",
                save_stub=f"cell06c_GLOBAL_gender_{_tag}",
            )
            results_json["editorial_trends"]["GLOBAL"][f"gender_{_tag}"] = _ent

# %% [markdown]
# ## Cell 7 — Praat vs OpenSMILE: Systematic Correlation Table + Heatmap
#
# Auto-discovers OpenSMILE column names, computes Pearson r + Spearman r for
# comparable feature pairs. Prints a table and shows a Spearman r heatmap.

# %%
from scipy.stats import pearsonr, spearmanr as _spearmanr

for _L in LANGS:
  print(f"\n=== Cell 7 · {_L} · Praat vs OpenSMILE correlation table ===")
  _praat_tsv   = idir / f"{_L}_praat.tsv"
  _osmile_tsv  = idir / f"{_L}_opensmile.tsv"

  if not _praat_tsv.exists() or not _osmile_tsv.exists():
    print(f"[{_L}] MISSING TSVs: need {_praat_tsv.name} and {_osmile_tsv.name}"); continue

  _praat_df  = pd.read_csv(_praat_tsv,  sep="\t", nrows=TEST_RUN_N if TEST_RUN else None)
  _osmile_df = pd.read_csv(_osmile_tsv, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None)
  _merged    = _praat_df.merge(_osmile_df, on="utterance_id", how="inner")
  print(f"[{_L}] Merged rows: {len(_merged):,}  "
        f"(praat={len(_praat_df):,}, osmile={len(_osmile_df):,})")

  def find_osm_col(keywords: list[str], df: pd.DataFrame) -> str | None:
    cols = df.columns.tolist()
    for kw in keywords:
      hits = [c for c in cols if kw.lower() in c.lower()]
      if hits: return hits[0]
    return None

  _PAIRS = [
    ("f0_raw",         find_osm_col(["F0semitone", "F0semi"], _osmile_df), "F0 (Hz vs semitone)"),
    ("intensity_raw",  find_osm_col(["Loudness_sma3", "loudness"], _osmile_df), "Intensity / Loudness"),
    ("f1_median",      find_osm_col(["F1frequency"], _osmile_df), "F1 frequency"),
    ("f2_median",      find_osm_col(["F2frequency"], _osmile_df), "F2 frequency"),
    ("f3_median",      find_osm_col(["F3frequency"], _osmile_df), "F3 frequency"),
    ("hnr_utt",        find_osm_col(["HNRdBACF", "HNR"], _osmile_df), "HNR"),
    ("jitter_local",   find_osm_col(["jitterLocal", "jitter"], _osmile_df), "Jitter"),
    ("shimmer_local",  find_osm_col(["shimmerLocal", "shimmer"], _osmile_df), "Shimmer"),
  ]

  rows_tab = []
  for praat_col, osm_col, label in _PAIRS:
    if praat_col not in _merged.columns or osm_col is None or osm_col not in _merged.columns:
      rows_tab.append({"Feature": label, "Praat": praat_col,
                       "OpenSMILE": str(osm_col), "Pearson r": None,
                       "Spearman r": None, "n": None}); continue
    sub = _merged[[praat_col, osm_col]].dropna()
    if len(sub) < 10:
      rows_tab.append({"Feature": label, "Praat": praat_col,
                       "OpenSMILE": osm_col, "Pearson r": None,
                       "Spearman r": None, "n": len(sub)}); continue
    pr, _ = pearsonr(sub[praat_col], sub[osm_col])
    sr, _ = _spearmanr(sub[praat_col], sub[osm_col])
    rows_tab.append({"Feature": label, "Praat": praat_col,
                     "OpenSMILE": osm_col,
                     "Pearson r": round(pr, 3), "Spearman r": round(sr, 3),
                     "n": len(sub)})

  _tab_df = pd.DataFrame(rows_tab)
  print(f"\n[{_L}] Praat vs OpenSMILE correlation table:")
  print(_tab_df.to_string(index=False))
  results_json.setdefault("praat_vs_opensmile", {})[_L] = [
    {k: v for k, v in row.items()} for row in rows_tab
  ]

  # Heatmap (Spearman r)
  _valid = _tab_df.dropna(subset=["Spearman r"])
  if _valid.empty:
    print(f"[{_L}] No valid pairs to plot."); continue

  fig, ax = plt.subplots(figsize=(max(6, len(_valid) * 1.1 + 2), 2.4))
  _mat = _valid[["Spearman r"]].values.T.astype(float)
  im = ax.imshow(_mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
  ax.set_xticks(range(len(_valid)))
  ax.set_xticklabels(_valid["Feature"].tolist(), rotation=25, ha="right", fontsize=9)
  ax.set_yticks([0]); ax.set_yticklabels(["Spearman r"], fontsize=9)
  for j, val in enumerate(_mat[0]):
    ax.text(j, 0, f"{val:.2f}", ha="center", va="center", fontsize=9,
            color="black" if abs(val) < 0.7 else "white")
  plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
  ax.set_title(f"Praat vs OpenSMILE — {_L}", fontsize=11)
  plt.tight_layout()
  save_fig(f"cell07_praat_vs_opensmile_heatmap_{_L}")

# %% [markdown]
# ## Cell 8 — Sentiment vs VAD: Binned Mean + Median, Spearman r
#
# Bins utterances by sentiment score, computes mean AND median VAD per bin.
# Shows 3-panel plot (valence, arousal, dominance) with both curves + linear fit.

# %%
_SENT_MIN, _SENT_MAX = 0.0, 5.0
_bins = np.linspace(_SENT_MIN, _SENT_MAX, N_BINS + 1)
_bin_centres = 0.5 * (_bins[:-1] + _bins[1:])
_vad_dims = ["valence", "arousal", "dominance"]
_all_vad_dfs = []   # collected per-lang for the global merged view
results_json.setdefault("sentiment_vs_vad", {})

# ── PER-LANGUAGE ───────────────────────────────────────────────────────────
for _L in LANGS:
  print(f"\n=== Cell 8 · {_L} · Sentiment × VAD ===")
  _vad_tsv = idir / f"{_L}_vad.tsv"
  if not _vad_tsv.exists():
    print(f"[{_L}] MISSING {_vad_tsv.name}. Run 35_vad.py first."); continue

  _vad_df = pd.read_csv(_vad_tsv, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None)
  _vad_df["_lang"] = _L
  _all_vad_dfs.append(_vad_df)
  print(f"[{_L}] VAD rows: {len(_vad_df):,},  "
        f"coverage: {_vad_df['valence'].notna().sum():,} utterances with ≥1 lemma matched "
        f"({100*_vad_df['valence'].notna().sum()/len(_vad_df):.1f}%)")

  _vad_df["_bin"] = pd.cut(_vad_df["sentiment_score"], bins=_bins, labels=False, include_lowest=True)

  fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
  for ax, dim in zip(axes, _vad_dims):
    sub = _vad_df.dropna(subset=["sentiment_score", dim])
    if len(sub) < 10:
      ax.set_title(f"{dim} — insufficient data"); continue
    sr, sp = _spearmanr(sub["sentiment_score"], sub[dim])
    _grouped = sub.groupby("_bin")[dim]
    _means   = _grouped.mean().reindex(range(N_BINS))
    _medians = _grouped.median().reindex(range(N_BINS))
    _counts  = _grouped.count().reindex(range(N_BINS), fill_value=0)
    _ok = _counts >= 3
    ax.plot(_bin_centres[_ok], _means[_ok].values, "o-", ms=4, lw=1.5, color="#2196F3", label="bin mean")
    ax.plot(_bin_centres[_ok], _medians[_ok].values, "s--", ms=4, lw=1.5, color="#FF9800", label="bin median")
    _x_fit = _bin_centres[_ok]; _y_fit = _means[_ok].values
    if len(_x_fit) >= 3:
      _m, _b = np.polyfit(_x_fit, _y_fit, 1)
      ax.plot(_x_fit, _m * _x_fit + _b, "k:", lw=1.2, label=f"slope={_m:.3f}")
      logprint(f"[{_L}] linear-fit slope · {dim}", f"{_m:+.4f} (intercept {_b:+.4f})", indent=1)
    ax.set_title(f"{dim}\nSpearman r={sr:.3f}, p={sp:.3e}, n={len(sub):,}", fontsize=10)
    ax.set_xlabel("Sentiment score", fontsize=9)
    ax.set_ylabel(dim.capitalize(), fontsize=9)
    ax.legend(fontsize=8)
    logprint(f"[{_L}] Spearman sentiment × {dim}", f"ρ={sr:+.4f} p={sp:.3e} n={len(sub):,}", indent=1)
    results_json["sentiment_vs_vad"].setdefault(_L, {})[dim] = {
        "rho": float(sr), "p": float(sp), "n": int(len(sub))
    }

  plt.suptitle(f"Sentiment vs VAD — {_L}", fontsize=12, y=1.01)
  plt.tight_layout()
  save_fig(f"cell08_sentiment_vs_vad_{_L}")

  # Mean-median agreement table (per lang)
  print(f"[{_L}] Bin mean vs median agreement:")
  _agree_rows = []
  for dim in _vad_dims:
    sub = _vad_df.dropna(subset=["sentiment_score", dim])
    _g = sub.groupby("_bin")[dim]
    _diff = (_g.mean().reindex(range(N_BINS)) - _g.median().reindex(range(N_BINS))).abs()
    _agree_rows.append({"dim": dim,
                        "mean |mean-median|": round(_diff.mean(), 4),
                        "max |mean-median|":  round(_diff.max(), 4)})
  print(pd.DataFrame(_agree_rows).to_string(index=False))

# ── GLOBAL: MERGED ACROSS LANGUAGES ────────────────────────────────────────
if len(_all_vad_dfs) >= 2:
  print(f"\n=== Cell 8 · GLOBAL (all {len(_all_vad_dfs)} langs merged) · Sentiment × VAD ===")
  _vad_g = pd.concat(_all_vad_dfs, ignore_index=True)
  _vad_g["_bin"] = pd.cut(_vad_g["sentiment_score"], bins=_bins, labels=False, include_lowest=True)
  print(f"[GLOBAL] Total VAD rows: {len(_vad_g):,} across {_vad_g['_lang'].nunique()} langs")

  fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
  for ax, dim in zip(axes, _vad_dims):
    sub = _vad_g.dropna(subset=["sentiment_score", dim])
    if len(sub) < 10:
      ax.set_title(f"{dim} — insufficient data"); continue
    sr, sp = _spearmanr(sub["sentiment_score"], sub[dim])
    logprint(f"[GLOBAL] Spearman sentiment × {dim}", f"ρ={sr:+.4f} p={sp:.3e} n={len(sub):,}", indent=1)
    results_json["sentiment_vs_vad"].setdefault("GLOBAL", {})[dim] = {
        "rho": float(sr), "p": float(sp), "n": int(len(sub))
    }
    _grouped = sub.groupby("_bin")[dim]
    _means   = _grouped.mean().reindex(range(N_BINS))
    _counts  = _grouped.count().reindex(range(N_BINS), fill_value=0)
    _ok = _counts >= 3
    ax.plot(_bin_centres[_ok], _means[_ok].values, "-", lw=3.0, color="#FFA500")
    _x_fit = _bin_centres[_ok]; _y_fit = _means[_ok].values
    if len(_x_fit) >= 3:
      _m, _b = np.polyfit(_x_fit, _y_fit, 1)
      ax.plot(_x_fit, _m * _x_fit + _b, "k:", lw=1.2, label=f"slope={_m:.3f}")
      logprint(f"[GLOBAL] linear-fit slope · {dim}", f"{_m:+.4f} (intercept {_b:+.4f})", indent=1)
    ax.set_title(f"{dim} (GLOBAL)\nSpearman r={sr:.3f}, p={sp:.3e}, n={len(sub):,}", fontsize=10)
    ax.set_xlabel("Sentiment score", fontsize=9)
    ax.set_ylabel(dim.capitalize(), fontsize=9)
    ax.legend(fontsize=8)

  plt.suptitle(f"Sentiment vs VAD — GLOBAL (all langs merged)", fontsize=12, y=1.01)
  plt.tight_layout()
  save_fig("cell08_sentiment_vs_vad_GLOBAL")

  # Per-language + global summary table
  print(f"\n[GLOBAL vs per-lang] Sentiment × VAD Spearman ρ table:")
  _rho_rows = []
  for dim in _vad_dims:
    _row = {"dim": dim}
    for _L in LANGS:
      _sub_L = _vad_g[(_vad_g["_lang"] == _L)].dropna(subset=["sentiment_score", dim])
      if len(_sub_L) >= 3:
        _r, _p = _spearmanr(_sub_L["sentiment_score"], _sub_L[dim])
        _row[_L] = f"{_r:+.3f} (n={len(_sub_L):,})"
      else:
        _row[_L] = "n<3"
    _sub_g = _vad_g.dropna(subset=["sentiment_score", dim])
    _r, _p = _spearmanr(_sub_g["sentiment_score"], _sub_g[dim])
    _row["GLOBAL"] = f"{_r:+.3f} (n={len(_sub_g):,})"
    _rho_rows.append(_row)
  print(pd.DataFrame(_rho_rows).to_string(index=False))

# %% [markdown]
# ## Cell 9 — CAP Topic Correlation
#
# Auto-discovers the parliamentary topic annotation field in the v4 JSONL.
# Per language: one-way ANOVA of acoustic features by topic.
# Global view: z-score normalise within each language, average by topic across langs.
# MIN_TOPIC_N = 50 utterances threshold for inclusion.

# %%
MIN_TOPIC_N = 50
_ALL_FEATS = (cfg["analysis"].get("features_main", [])
              + cfg["analysis"].get("features_appendix", []))
if not _ALL_FEATS:
    _ALL_FEATS = ["f0_raw", "intensity_norm", "speechrate_wps",
                  "f0_norm", "intensity_raw", "speechrate_sps"]
_FEAT_FOR_TOPIC = _ALL_FEATS  # or set to a subset, e.g. ["f0_raw", "speechrate_wps"]
_JSONL_DIR = Path(cfg["paths"]["data_root"])

def _discover_topic_field(jsonl_path: Path, n_probe: int = 200) -> str | None:
    """Probe first n_probe records and return the first field whose name suggests a topic."""
    _keywords = ["topic", "cap", "category", "subject", "policy", "issue"]
    _candidates: dict[str, int] = {}
    with open(jsonl_path, encoding="utf-8") as _f:
        for i, _line in enumerate(_f):
            if i >= n_probe:
                break
            try:
                _rec = json.loads(_line)
            except json.JSONDecodeError:
                continue
            # Flatten one level
            _flat = {}
            for _k, _v in _rec.items():
                if isinstance(_v, dict):
                    for _kk, _vv in _v.items():
                        _flat[f"{_k}.{_kk}"] = _vv
                else:
                    _flat[_k] = _v
            for _fk, _fv in _flat.items():
                if any(_kw in _fk.lower() for _kw in _keywords):
                    if isinstance(_fv, (str, int, float)) and _fv not in (None, ""):
                        _candidates[_fk] = _candidates.get(_fk, 0) + 1
    if not _candidates:
        return None
    return max(_candidates, key=lambda k: _candidates[k])


def _load_topic_map(jsonl_path: Path, target_ids: set, field: str) -> dict[str, str]:
    """Stream JSONL and return {utterance_id: topic_value} for target_ids."""
    _top_keys = field.split(".", 1)
    _result = {}
    with open(jsonl_path, encoding="utf-8") as _f:
        for _line in _f:
            if len(_result) >= len(target_ids):
                break
            _line = _line.strip()
            if not _line:
                continue
            try:
                _rec = json.loads(_line)
            except json.JSONDecodeError:
                continue
            _uid = _rec.get("id")
            if _uid not in target_ids:
                continue
            if len(_top_keys) == 1:
                _val = _rec.get(_top_keys[0])
            else:
                _val = (_rec.get(_top_keys[0]) or {}).get(_top_keys[1])
            if _val is not None:
                _result[_uid] = str(_val)
    return _result


from scipy.stats import f_oneway as _f_oneway

# Prefer topic from filtered JSONL (10_filter.py v1.02+ preserves it).
# Fall back to raw v4 JSONL discovery only when filtered has no topic column.
_filt_has_topic = False
_probe_filt = idir / f"{LANG}_filtered.jsonl"
if _probe_filt.exists():
    with open(_probe_filt) as _pf:
        _first_line = next((l for l in _pf if l.strip()), None)
    if _first_line:
        _first_rec = json.loads(_first_line)
        _filt_has_topic = _first_rec.get("topic") is not None

_topic_field = "topic" if _filt_has_topic else None
if _filt_has_topic:
    print(f"Using 'topic' column from filtered JSONL (no raw v4 access needed).")
else:
    for _lang in [LANG] + [l for l in ALL_LANGS if l != LANG]:
        _v4_path = _JSONL_DIR / f"ParlaSpeech-{_lang}.v4.0.patched.jsonl"
        if _v4_path.exists():
            _topic_field = _discover_topic_field(_v4_path)
            if _topic_field:
                print(f"Filtered JSONL missing topic; auto-discovered '{_topic_field}' from raw v4 ({_lang}).")
                break

if _topic_field is None:
    print("No topic field found (rerun 10_filter.py v1.02+ to preserve topic). Skipping CAP cell.")
else:
    # --- Per-language ANOVA ---
    print(f"\n--- Per-language ANOVA: features by topic (field='{_topic_field}') ---")
    _all_lang_topic_dfs = {}  # lang → merged df with topic column

    for _lang in ALL_LANGS:
        _v4_path  = _JSONL_DIR / f"ParlaSpeech-{_lang}.v4.0.patched.jsonl"
        _feat_tsv = idir / f"{_lang}_praat.tsv"
        _filt_jsn = idir / f"{_lang}_filtered.jsonl"
        # Filtered JSONL is required; raw v4 only if we're falling back
        if not (_feat_tsv.exists() and _filt_jsn.exists()):
            print(f"[{_lang}] Missing files, skipping.")
            continue
        if not _filt_has_topic and not _v4_path.exists():
            print(f"[{_lang}] Missing raw v4 (fallback source), skipping.")
            continue

        if TEST_RUN:
            _filt_recs = []
            with open(_filt_jsn) as _fjs:
                for _l in _fjs:
                    if len(_filt_recs) >= TEST_RUN_N: break
                    if _l.strip(): _filt_recs.append(json.loads(_l))
        else:
            _filt_recs = [json.loads(l) for l in open(_filt_jsn) if l.strip()]

        if _filt_has_topic:
            _topic_map = {r["utterance_id"]: str(r["topic"])
                          for r in _filt_recs if r.get("topic") is not None}
        else:
            _target_ids = {r["utterance_id"] for r in _filt_recs}
            _topic_map  = _load_topic_map(_v4_path, _target_ids, _topic_field)

        _feat_df = pd.read_csv(_feat_tsv, sep="\t", nrows=TEST_RUN_N if TEST_RUN else None)
        _feat_df["_topic"] = _feat_df["utterance_id"].map(_topic_map)
        _feat_df = _feat_df.dropna(subset=["_topic"])

        _topic_counts = _feat_df["_topic"].value_counts()
        _keep_topics  = _topic_counts[_topic_counts >= MIN_TOPIC_N].index
        _feat_df      = _feat_df[_feat_df["_topic"].isin(_keep_topics)]
        print(f"[{_lang}] {len(_keep_topics)} topics (≥{MIN_TOPIC_N} utts), "
              f"{len(_feat_df):,} utterances retained")

        _all_lang_topic_dfs[_lang] = _feat_df

        # ANOVA per feature
        _anova_rows = []
        for _feat in _FEAT_FOR_TOPIC:
            if _feat not in _feat_df.columns:
                continue
            _groups = [_feat_df[_feat_df["_topic"] == t][_feat].dropna().values
                       for t in _keep_topics]
            _groups = [g for g in _groups if len(g) >= 3]
            if len(_groups) < 2:
                continue
            _F, _p = _f_oneway(*_groups)
            _anova_rows.append({"feature": _feat, "F": round(_F, 2), "p": round(_p, 5),
                                 "n_topics": len(_groups)})
        if _anova_rows:
            _anov_df = pd.DataFrame(_anova_rows).sort_values("p")
            print(f"  ANOVA (top features):\n{_anov_df.head(5).to_string(index=False)}")

    # --- Per-language bar chart for LANG ---
    if LANG in _all_lang_topic_dfs:
        _df_t = _all_lang_topic_dfs[LANG]
        _feat_for_plot = [f for f in _FEAT_FOR_TOPIC if f in _df_t.columns][:4]
        _n_f = len(_feat_for_plot)
        if _n_f > 0:
            fig, axes = plt.subplots(1, _n_f, figsize=(5 * _n_f, 5), sharey=False)
            if _n_f == 1:
                axes = [axes]
            for _ax, _feat in zip(axes, _feat_for_plot):
                _topic_means = _df_t.groupby("_topic")[_feat].mean().sort_values()
                _topic_sems  = _df_t.groupby("_topic")[_feat].sem().reindex(_topic_means.index)
                _ax.barh(_topic_means.index, _topic_means.values,
                         xerr=_topic_sems.values, capsize=3, color="#5C85D6")
                _ax.set_title(FEAT_LABELS.get(_feat, _feat), fontsize=10)
                _ax.set_xlabel("Mean feature value", fontsize=9)
                _ax.tick_params(axis="y", labelsize=7)
            plt.suptitle(f"Feature means by CAP topic — {LANG}", fontsize=12)
            plt.tight_layout()
            save_fig(f"cell09_topic_means_{LANG}")

    # --- Global z-score normalised view ---
    print("\n--- Global z-score normalised means by topic ---")
    _global_rows = []
    for _lang, _df_t in _all_lang_topic_dfs.items():
        for _feat in _FEAT_FOR_TOPIC:
            if _feat not in _df_t.columns:
                continue
            _mu, _sd = _df_t[_feat].mean(), _df_t[_feat].std()
            if _sd < 1e-9:
                continue
            _df_t = _df_t.copy()
            _df_t[f"_z_{_feat}"] = (_df_t[_feat] - _mu) / _sd
            for _top in _df_t["_topic"].unique():
                _sub = _df_t[_df_t["_topic"] == _top][f"_z_{_feat}"].dropna()
                if len(_sub) < MIN_TOPIC_N:
                    continue
                _global_rows.append({"lang": _lang, "topic": _top,
                                     "feature": _feat, "z_mean": _sub.mean()})

    if _global_rows:
        _glob_df  = pd.DataFrame(_global_rows)
        _pivot    = _glob_df.groupby(["topic", "feature"])["z_mean"].mean().unstack("feature")
        _pivot    = _pivot.dropna(how="all")
        _feat_cols = [f for f in _FEAT_FOR_TOPIC if f in _pivot.columns]
        if _feat_cols:
            _pivot = _pivot[_feat_cols]
            _order = _pivot.mean(axis=1).sort_values().index
            _pivot = _pivot.loc[_order]
            fig, ax = plt.subplots(figsize=(max(6, len(_feat_cols) * 2), max(4, len(_pivot) * 0.45 + 1)))
            im2 = ax.imshow(_pivot.values.T, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
            ax.set_xticks(range(len(_pivot)))
            ax.set_xticklabels(_pivot.index.tolist(), rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(len(_feat_cols)))
            ax.set_yticklabels([FEAT_LABELS.get(f, f) for f in _feat_cols], fontsize=9)
            plt.colorbar(im2, ax=ax, shrink=0.6, label="z-score")
            ax.set_title("Global z-score normalised feature means by CAP topic", fontsize=11)
            plt.tight_layout()
            save_fig("cell09_topic_global_zscore")
    else:
        print("No global rows — check topic field coverage.")

# %% [markdown]
# ## Cell 10 — Read + display the pipeline results (results/*.json)
#
# Reads the JSONs produced by 3_analysis/ and 4_outputs/41_numbers.py and prints
# a compact summary of the paper's headline numbers so the notebook is a single
# entry point for "what does the pipeline say right now". Absent JSONs are
# skipped with a note. Everything read here is also mirrored into
# results.json["pipeline"] so the notebook's own JSON contains a copy.

# %%
_res_dir = (_repo_root / cfg["paths"]["results_dir"]).resolve()
print(f"\n=== Cell 10 · Pipeline results in {_res_dir} ===")

_pipeline: dict = {}
_files = {
    "numbers":         _res_dir / "numbers.json",
    "h1":              _res_dir / "h1_results.json",
    "h2":              _res_dir / "h2_results.json",
    "h3":              _res_dir / "h3_results.json",
    "vad_correlations":_res_dir / "vad_correlations.json",
    "global_trend":    _res_dir / "global_trend.json",
}
for _k, _p in _files.items():
    if _p.exists():
        with open(_p) as _fh:
            _pipeline[_k] = json.load(_fh)
        print(f"  [OK] {_p.name} — {_p.stat().st_size:,} bytes")
    else:
        print(f"  [--] {_p.name} not present (run 3_analysis / 4_outputs first)")

results_json["pipeline"] = _pipeline

# numbers.json headline
if "numbers" in _pipeline:
    _num = _pipeline["numbers"]
    print(f"\n[numbers.json] Headline hypothesis counts:")
    for _k in ["h1_sig_speaker_avg", "h1_total", "h2_sig", "h2_total",
               "h3_strong", "h3_partial", "h3_supported", "h3_total"]:
        if _k in _num:
            print(f"  {_k:<28s} {_num[_k]}")

# H1 per lang×feature — p_bh + rank-biserial
if "h1" in _pipeline:
    _h1_rows = []
    for _key, _v in _pipeline["h1"].items():
        _sa = _v.get("speaker_avg", {})
        _h1_rows.append({"key": _key,
                         "n": _sa.get("n"),
                         "RBC": round(_sa.get("rbc", float("nan")), 3),
                         "p_bh": _sa.get("p_bh"),
                         "concord": round(_sa.get("concordance", float("nan")), 3)})
    _h1_df = pd.DataFrame(_h1_rows).sort_values("p_bh")
    print(f"\n[H1 — Wilcoxon speaker-avg] {len(_h1_df)} tests, ranked by p_bh:")
    print(_h1_df.head(15).to_string(index=False))
    _sig = _h1_df[_h1_df["p_bh"] < 0.05]
    print(f"  {len(_sig)}/{len(_h1_df)} significant at p_bh < 0.05")

# H2 per lang×feature — mean_tau + CI + significant-speaker fraction
if "h2" in _pipeline:
    _h2 = _pipeline["h2"]
    print(f"\n[H2 — Kendall τ] {len(_h2)} tests, ranked by |mean_tau|:")
    _h2_rows = []
    for _key, _v in _h2.items():
        if not isinstance(_v, dict): continue
        _mt = _v.get("mean_tau", float("nan"))
        _ns, _nt = _v.get("n_sig_speakers", 0), _v.get("n_speakers", 1) or 1
        _h2_rows.append({"key": _key,
                         "mean_tau": round(_mt, 3),
                         "ci_lo":    round(_v.get("ci_lo", float("nan")), 3),
                         "ci_hi":    round(_v.get("ci_hi", float("nan")), 3),
                         "sig_spk":  f"{_ns}/{_nt} ({100*_ns/_nt:.0f}%)",
                         "p_bh":     _v.get("p_bh")})
    _h2_df = pd.DataFrame(_h2_rows)
    _h2_df = _h2_df.reindex(_h2_df["mean_tau"].abs().sort_values(ascending=False).index)
    print(_h2_df.head(15).to_string(index=False))

# VAD correlations from pipeline
if "vad_correlations" in _pipeline:
    print(f"\n[VAD pipeline results]  (compare with Cell 8's per-lang/GLOBAL Spearman)")
    print(json.dumps(_pipeline["vad_correlations"], indent=2)[:1500])
    print("  ... (truncated; full copy is in results.json)")

print(f"\n[results.json] Everything above + Cell 8/6c/1b metrics are being dumped at exit.")
