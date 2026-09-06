#!/usr/bin/env python3
# ============================================================
# Script:  50_explore.py  (Jupyter-compatible notebook)
# Release: 1.0
# Version: v1.00
# Purpose: Debug and exploration for ParlaSpeech sentiment-acoustics.
#          Envelope viewer, speechrate+transcript, general replotters,
#          Praat vs OpenSMILE comparison, per-language anomaly inspector.
#
# Usage:   Open in VS Code with Jupyter extension (cells = # %% blocks)
#          Convert: jupytext --to notebook 50_explore.py
#          Run from: JP1_parlaspeech-sentiment-acoustics/ directory
# ============================================================

# %% [markdown]
# # ParlaSpeech Sentiment-Acoustics — Exploration Notebook
#
# Cells can be run independently after the Setup cell.
# Adjust `LANG`, `SEED`, `N_EXAMPLES`, `REROLL` in Setup.

# %% Setup — run this first
import sys
import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

sys.path.insert(0, str(Path(".").resolve()))
from utils.config_loader import load_config, get_intermediate_dir, get_results_dir
from utils.data_utils import load_jsonl

# ── Configure here ──────────────────────────────────────────────────────────
LANG    = "SI"    # HR, CZ, PL, RS, SI
SEED    = 42      # reproducible sample; ignored when REROLL=True
N_EXAMPLES = 6    # utterances to display per cell
REROLL  = False   # True = new random sample each run
# ────────────────────────────────────────────────────────────────────────────

rng = random.Random(None if REROLL else SEED)

cfg   = load_config("config.json")
idir  = get_intermediate_dir(cfg)
rdir  = get_results_dir(cfg)

PALETTE = {
    "HR": "#E63946", "CZ": "#2A9D8F", "PL": "#E9C46A",
    "RS": "#264653", "SI": "#A8DADC",
}

# Load features TSV for LANG (if present)
feats_path = idir / f"{LANG}_features.tsv"
df_feats = pd.read_csv(feats_path, sep="\t") if feats_path.exists() else pd.DataFrame()
if df_feats.empty:
    print(f"[WARN] {feats_path} not found — feature columns unavailable in some cells.")
else:
    print(f"Loaded {len(df_feats):,} utterances from {feats_path.name}")
    print(f"  Columns: {list(df_feats.columns[:12])} ...")

# %% [markdown]
# ## Cell 2 — Envelope Viewer (Praat NPZ)
#
# Loads `{LANG}_praat_envelopes.npz` and plots N random utterances.
# Each shows: F0 track, Intensity track, per-word F1/F2/F3 medians.
# Word boundaries are drawn as vertical lines.

# %%
npz_path = idir / f"{LANG}_praat_envelopes.npz"

if not npz_path.exists():
    print(f"[SKIP] {npz_path} not found. Run 20_extract_praat.py first.")
else:
    data = np.load(npz_path, allow_pickle=True)
    uids = list(data["utterance_ids"])
    uid2idx = {uid: i for i, uid in enumerate(uids)}

    # Build a lookup from features TSV
    uid2meta = {}
    if not df_feats.empty:
        for _, row in df_feats.iterrows():
            uid2meta[row["utterance_id"]] = row

    # Sample N utterances that have F0 data
    candidates = [uid for uid in uids if data["f0_values"][uid2idx[uid]].size > 0]
    sample_uids = rng.sample(candidates, min(N_EXAMPLES, len(candidates)))

    for uid in sample_uids:
        i = uid2idx[uid]
        f0_t  = data["f0_times"][i].astype(float)
        f0_v  = data["f0_values"][i].astype(float)
        int_t = data["intensity_times"][i].astype(float)
        int_v = data["intensity_values"][i].astype(float)
        w_s   = data["word_starts"][i].astype(float)
        w_e   = data["word_ends"][i].astype(float)
        f1_w  = data["f1_word_median"][i].astype(float)
        f2_w  = data["f2_word_median"][i].astype(float)
        f3_w  = data["f3_word_median"][i].astype(float)

        meta = uid2meta.get(uid, {})
        sent  = meta.get("sentiment_score", "?")
        label = meta.get("sentiment_label", "?")
        n_w   = meta.get("n_words", "?")
        f0_sc = meta.get("f0_raw", "?")
        spk   = meta.get("speaker_id", "?")

        fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=False)
        fig.suptitle(
            f"{uid}  |  spk={spk}  sent={sent:.2f} ({label})  n_words={n_w}  f0_raw={f0_sc}",
            fontsize=9, y=1.01
        )

        # F0 envelope
        ax = axes[0]
        f0_valid = ~np.isnan(f0_v)
        if f0_valid.any():
            ax.plot(f0_t[f0_valid], f0_v[f0_valid], color="#E63946", lw=1.2, label="F0")
        for s, e in zip(w_s, w_e):
            ax.axvline(s, color="gray", lw=0.5, alpha=0.4)
        ax.set_ylabel("F0 (Hz)")
        ax.set_title("F0 envelope")

        # Intensity envelope
        ax = axes[1]
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
        plt.show()

# %% [markdown]
# ## Cell 2b — VAD Word Envelope Viewer
#
# Loads `{LANG}_vad_envelopes.npz` and plots per-word valence/arousal/dominance
# for N random utterances. NaN = word not in lexicon (non-content UPOS or OOV).
# Aligned to word timing from v4 words[], same x-axis as F0 envelopes.

# %%
vad_npz_path = idir / f"{LANG}_vad_envelopes.npz"
vad_tsv_path = idir / f"{LANG}_vad.tsv"

if not vad_npz_path.exists():
    print(f"[SKIP] {vad_npz_path} not found. Run 35_vad.py with save_vad_envelopes=true.")
else:
    vdata = np.load(vad_npz_path, allow_pickle=True)
    vad_uids = list(vdata["utterance_ids"])
    vuid2idx = {uid: i for i, uid in enumerate(vad_uids)}

    # Load utterance-level VAD scores for metadata
    df_vad = pd.read_csv(vad_tsv_path, sep="\t") if vad_tsv_path.exists() else pd.DataFrame()
    uid2vad = {}
    if not df_vad.empty:
        for _, row in df_vad.iterrows():
            uid2vad[row["utterance_id"]] = row

    # Sample utterances with ≥1 covered word
    candidates_v = [
        uid for uid in vad_uids
        if np.any(~np.isnan(vdata["word_valences"][vuid2idx[uid]].astype(float)))
    ]
    sample_v_uids = rng.sample(candidates_v, min(N_EXAMPLES, len(candidates_v)))

    for uid in sample_v_uids:
        i = vuid2idx[uid]
        starts    = vdata["word_starts"][i].astype(float)
        ends      = vdata["word_ends"][i].astype(float)
        valences  = vdata["word_valences"][i].astype(float)
        arousals  = vdata["word_arousals"][i].astype(float)
        dominances = vdata["word_dominances"][i].astype(float)

        meta = uid2meta.get(uid, {})
        vad_meta = uid2vad.get(uid, {})
        sent  = meta.get("sentiment_score", "?")
        utt_v = vad_meta.get("valence", "?")
        utt_a = vad_meta.get("arousal", "?")
        n_cov = int(vad_meta.get("vad_n_covered", 0))
        n_w   = len(starts)

        fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
        fig.suptitle(
            f"{uid}  |  sent={sent:.2f}  utt_valence={utt_v:.3f}  "
            f"utt_arousal={utt_a:.3f}  covered={n_cov}/{n_w} words",
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
        plt.show()


# %% [markdown]
# ## Cell 3 — Speechrate + Transcript Viewer
#
# For each sampled utterance: word timeline bars (colored by duration),
# silent_pauses as shaded regions, transcript text with vowels marked.

# %%
jsonl_path = idir / f"{LANG}_filtered.jsonl"

# Vowel sets per language
VOWELS = {
    "HR": set("aeiouAEIOUčšžćđČŠŽĆĐ"),
    "CZ": set("aeiouáéíóúůýAEIOUÁÉÍÓÚŮÝ"),
    "PL": set("aeiouąęóAEIOUĄĘÓ"),
    "RS": set("aeiouAEIOUčšžćđČŠŽĆĐ"),
    "SI": set("aeiouAEIOUčšžČŠŽ"),
}
lang_vowels = VOWELS.get(LANG, set("aeiouAEIOU"))

if not jsonl_path.exists():
    print(f"[SKIP] {jsonl_path} not found.")
else:
    records = load_jsonl(jsonl_path)
    # Filter to utterances that have speechrate in features and word timing
    has_words = [r for r in records if r.get("words_align")]
    if not df_feats.empty:
        sr_uids = set(df_feats.dropna(subset=["speechrate_wps"])["utterance_id"])
        has_words = [r for r in has_words if r["utterance_id"] in sr_uids]

    sample_recs = rng.sample(has_words, min(N_EXAMPLES, len(has_words)))

    for rec in sample_recs:
        uid   = rec["utterance_id"]
        text  = rec.get("text", "")
        words = rec.get("words_align", [])
        pauses = rec.get("silent_pauses") or []
        meta  = uid2meta.get(uid, {}) if not df_feats.empty else {}
        sr    = meta.get("speechrate_wps", "?")
        sent  = meta.get("sentiment_score", "?")
        n_paus = len(pauses)

        if not words:
            continue

        # Timeline plot
        fig, ax = plt.subplots(figsize=(14, 2.5))
        durations = [w.get("end", 0) - w.get("start", 0) for w in words]
        max_dur = max(durations) if durations else 1
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

        for p in pauses:
            ps = p.get("time_s", p.get("start", 0))
            pe = p.get("time_e", p.get("end", ps))
            ax.axvspan(ps, pe, color="lightblue", alpha=0.5, label="Silent pause")

        ax.set_yticks([])
        ax.set_xlabel("Time (s)")
        ax.set_title(
            f"{uid}  |  sent={sent:.2f}  sr={sr:.2f} wps  n_silent_pauses={n_paus}",
            fontsize=9,
        )
        plt.tight_layout()
        plt.show()

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
    df = pd.read_csv(fp, sep="\t")
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
plt.show()

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
plt.show()

# %% [markdown]
# ## Cell 5 — Praat vs OpenSMILE Comparison
#
# Scatter-compares equivalent features from both extractors.
# Uses LLD NPZ if available (computes mean voiced F0), else scalar functionals TSV.

# %%
osm_path = idir / f"{LANG}_opensmile.tsv"
lld_path = idir / f"{LANG}_opensmile_lld.npz"

if not osm_path.exists() and not lld_path.exists():
    print(f"[SKIP] No OpenSMILE outputs found for {LANG}.")
elif df_feats.empty:
    print("[SKIP] Features TSV not loaded — run Setup cell first.")
else:
    comparison_pairs = []   # (praat_col, osm_col_or_lld_key, label)

    if lld_path.exists():
        lld = np.load(lld_path, allow_pickle=True)
        lld_uids = list(lld["utterance_ids"])
        uid2lld_idx = {uid: i for i, uid in enumerate(lld_uids)}

        # Compute per-utterance mean (voiced frames only) for F0 and loudness
        def mean_voiced(arr):
            v = arr.astype(float)
            v[v <= 0] = np.nan   # 0 = unvoiced sentinel in GeMAPSv01b
            return np.nanmean(v) if np.any(~np.isnan(v)) else np.nan

        f0_osm, loud_osm, f1_osm, f2_osm, f3_osm, uids_osm = [], [], [], [], [], []
        for uid, f0_arr, loud_arr, f1_arr, f2_arr, f3_arr in zip(
            lld_uids,
            lld["f0_lld"], lld["loudness_lld"],
            lld["f1_lld"], lld["f2_lld"], lld["f3_lld"],
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
        df_merged = df_feats.merge(df_osm, on="utterance_id", how="inner")

        pairs = [
            ("f0_raw",        "osm_f0",      "F0 raw vs OSM F0 (LLD mean voiced, log-semitones)"),
            ("intensity_norm","osm_loudness", "Intensity norm vs OSM Loudness (LLD mean)"),
            ("f1_median",     "osm_f1",       "F1 median vs OSM F1 (LLD mean voiced)"),
            ("f2_median",     "osm_f2",       "F2 median vs OSM F2 (LLD mean voiced)"),
            ("f3_median",     "osm_f3",       "F3 median vs OSM F3 (LLD mean voiced)"),
        ]
    else:
        df_osm = pd.read_csv(osm_path, sep="\t")
        df_merged = df_feats.merge(df_osm, on="utterance_id", how="inner")
        # Map approximate functional column names
        pairs = [
            ("f0_raw", "osmile_F0semitoneFrom27.5Hz_sma3nz_amean",
             "F0 raw vs OSM F0 (functional mean)"),
            ("intensity_norm", "osmile_Loudness_sma3_amean",
             "Intensity norm vs OSM Loudness (functional mean)"),
        ]

    valid_pairs = [(a, b, lbl) for a, b, lbl in pairs
                   if a in df_merged.columns and b in df_merged.columns]
    if not valid_pairs:
        print("[WARN] No matching column pairs found for comparison.")
    else:
        fig, axes = plt.subplots(1, len(valid_pairs), figsize=(len(valid_pairs) * 4, 4))
        if len(valid_pairs) == 1: axes = [axes]
        fig.suptitle(f"{LANG} — Praat vs OpenSMILE Feature Comparison", fontsize=11)

        for ax, (praat_col, osm_col, label) in zip(axes, valid_pairs):
            sub = df_merged[[praat_col, osm_col]].dropna()
            if len(sub) < 10:
                ax.set_title(f"{label}\n(n<10)")
                continue
            r, p = stats.spearmanr(sub[praat_col], sub[osm_col])
            ax.scatter(sub[praat_col], sub[osm_col],
                       alpha=0.15, s=8, color=PALETTE.get(LANG, "#888"))
            ax.set_xlabel(f"Praat: {praat_col}", fontsize=8)
            ax.set_ylabel(f"OSM: {osm_col.split('_')[-1]}", fontsize=8)
            ax.set_title(f"{label}\nSpearman r={r:.3f}, p={p:.3e}", fontsize=8)
            ax.tick_params(labelsize=7)

        plt.tight_layout()
        plt.show()

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
    df = pd.read_csv(fp, sep="\t")
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
plt.show()
