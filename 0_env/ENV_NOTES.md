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

## Praat

No separate Praat installation needed — parselmouth bundles it.
Praat version used: see `parselmouth.__version__` after install.

## Audio paths

Audio files are on gpu2. Set `config.json → paths.audio_root.{LANG}` before running
`2_features/20_extract_praat.py` and `2_features/22_extract_opensmile.py`.

Speech rate (`21_extract_speechrate.py`) does NOT need audio — only the JSONL timing data.

## NRC VAD Lexicon

Download from: https://saifmohammad.com/WebPages/nrclex.html
→ NRC-VAD-Lexicon.zip → NRC-VAD-Lexicon.txt

Set path in `config.json → vad.nrc_vad_path`.

The file includes ~20,000 English words with valence/arousal/dominance scores
plus translations to 100+ languages via automated translation.
Relevant column names for our languages:
  Croatian, Czech, Polish, Serbian, Slovene
(match to config.json → vad.nrc_lang_codes)
