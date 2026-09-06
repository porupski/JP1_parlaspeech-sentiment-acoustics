# Environment Notes

## Python env

Base: `parla_align4` conda env on tantra. Trimmed to `environment.yml` in this folder.

Key packages and why:
- `praat-parselmouth==0.4.5` — Praat via Python, F0 + intensity extraction
- `opensmile==2.5.1` — GeMAPS feature extraction (appendix track)
- `librosa` — audio loading utilities
- `rpy2==3.6.4` — R bridge for GAMM (disabled by default, see config.json)
- `scipy`, `statsmodels` — Wilcoxon, Kendall, BH correction
- `transformers`, `huggingface-hub` — needed if audio VAD model enabled
- `jupytext` — convert 42_plots_editorial.py → .ipynb

## R packages

Not managed by mamba. Install manually inside the env:

```r
install.packages(c("mgcv", "ordinal"))
```

R home is hardcoded in config.json → paths.r_home. Adjust to your env path.

## Setup

```bash
mamba env create -f 0_env/environment.yml
mamba activate jp1_ps_sent
```

## Notebooks

Exploration notebooks live in `5_debug/`. Each ships as both `.py` (jupytext
`# %%` format) and `.ipynb`. Open either in VS Code Jupyter extension directly.

To re-sync after editing the `.py` file:
```bash
jupytext --to notebook 5_debug/50_explore.py -o 5_debug/50_explore.ipynb
```

To sync changes made in the `.ipynb` back to `.py`:
```bash
jupytext --to py:percent 5_debug/50_explore.ipynb -o 5_debug/50_explore.py
```

## Praat

No separate Praat installation needed — parselmouth bundles it.
Praat version used: see `parselmouth.__version__` after install.

## Audio paths

Audio files are on gpu2. Set `config.json → paths.audio_root.{LANG}` before running
`2_features/20_extract_praat.py` and `2_features/22_extract_opensmile.py`.

Speech rate (`21_extract_speechrate.py`) does NOT need audio — only the JSONL timing data.

## NRC VAD Lexicon

Use **v1 (multilingual)** — NOT v2.1 (English-only). Run:

```bash
bash 0_env/get_nrc_vad.sh
```

This downloads from `https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip`
and extracts to `data/lexicons/NRC-VAD-Lexicon/`.

Due to the zip's internal structure, per-language files land at:
  `data/lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon/OneFilePerLanguage/`

This double-nested path is already set in `config.json → vad.nrc_vad_dir`.

File format (per language): `{LangName}-NRC-VAD-Lexicon.txt`
Columns (tab-separated): `English Word | Valence | Arousal | Dominance | Translated Word`
Lookup key = translated word (last column); VAD scores from English-annotated entry.

Language files used:
  HR → Croatian-NRC-VAD-Lexicon.txt
  CZ → Czech-NRC-VAD-Lexicon.txt
  PL → Polish-NRC-VAD-Lexicon.txt
  RS → Bosnian-NRC-VAD-Lexicon.txt  ← proxy! Serbian file is Cyrillic, corpus is Latin
  SI → Slovenian-NRC-VAD-Lexicon.txt

VAD lookup uses lemmas from v4 JSONL `linguistic_annotation` field (UPOS: NOUN/VERB/ADJ/ADV).
No separate lemmatizer needed.
