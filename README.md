# JP1 parlaspeech-sentiment-acoustics

This repository contains the analysis code for "From Vocal Cues to Political Views" (revision 1). The code measures acoustic correlates of sentiment in five Slavic parliamentary speech corpora.

**Languages:** HR · CZ · PL · RS · SI  
**Data:** ParlaSpeech 4.0 `.patched.jsonl`  
**Features:** Praat (F0, intensity) · speech rate · OpenSMILE GeMAPS (appendix)  
**Hypotheses:** H1 Wilcoxon extremes · H2 Kendall monotonic trend · H3 inflection-split

---

## Setup

### 1. Create the conda environment

```bash
mamba env create -f 0_env/environment.yml
mamba activate jp1_ps_sent
```

### 2. Download the NRC VAD Lexicon

```bash
bash 0_env/get_nrc_vad.sh
```

The script downloads NRC-VAD-Lexicon v2.1 and places the files in `data/lexicons/NRC-VAD-Lexicon-v2.1/`. The path is already set in `config.json → vad.nrc_vad_path`.

### 3. Install R packages (GAMM only — disabled by default)

Skip this step unless GAMM is needed. Set `enable_gamm: true` in `config.json` first.

```bash
Rscript -e 'install.packages(c("mgcv", "ordinal"))'
```

### 4. Edit config.json

Set the following values before running the pipeline:

- `paths.audio_root.{LANG}` — full path to the audio directory for each language on gpu2
- `paths.r_home` — path to the R installation (only required if GAMM is enabled)

### 5. Verify JSONL field names

Run the following check on the first use:

```bash
python -c "
import json
rec = json.loads(open('/cache/ivanp/projects/ParlaSpeech_v4/release_v4/ParlaSpeech-HR.v4.0.patched.jsonl').readline())
print(list(rec.keys()))
"
```

If the field names differ from `config.json → jsonl_fields`, update `config.json`. Do not edit the scripts.

---

## Running

```bash
# Speech rate extraction only — no audio files required
python run_all.py --skip praat opensmile --langs HR

# Full pipeline, all languages
python run_all.py

# Selected stages
python run_all.py --only h1 h2 h3
python run_all.py --skip gamm
```

Pipeline order:

```
filter → speechrate → praat → opensmile → normalize → join
→ h1 → h2 → h3 → corrections → vad → tables → numbers → plots
```

---

## Directory structure

```
config.json             All constants, paths, and flags. Edit here, not in scripts.
run_all.py              End-to-end pipeline runner.

0_env/                  Conda environment files and setup scripts.
1_data/                 Filters JSONL and joins feature TSVs.
2_features/             Extracts Praat, speech rate, and OpenSMILE features.
3_analysis/             Runs H1, H2, H3 tests and VAD correlations.
4_outputs/              Generates LaTeX tables, numbers.json, and figures.
utils/                  Shared library. Import from here; do not copy.
tests/                  Pytest unit tests.
archive/                Original scripts. Do not modify.
paper/                  LaTeX source.
results/                Generated tables, figures, and numbers.json.
```

---

## Outputs

| File | Contents |
|------|----------|
| `results/h1_results.json` | Wilcoxon p, RBC, concordance per language × feature |
| `results/h2_results.json` | Kendall τ̄, bootstrap CI, sig% per language × feature |
| `results/h3_results.json` | Split-side slopes and p per language × feature |
| `results/numbers.json` | All in-text paper numbers, keyed |
| `results/tables/*.tex` | LaTeX tables |
| `results/figures/*.png` | Figures 1–4 |

All in-text numbers in the LaTeX source come from `numbers.json`. No numbers are typed by hand.

---

## Config flags

| Flag | Default | Effect |
|------|---------|--------|
| `enable_gamm` | `false` | Runs GAMM via rpy2 (requires R packages) |
| `enable_opensmile` | `true` | Runs OpenSMILE GeMAPS extraction (appendix track) |
| `enable_text_vad` | `true` | Runs NRC VAD lexicon correlation |
| `enable_audio_vad` | `false` | Runs audio-based VAD model (GPU recommended) |
| `kendall_on_bins` | `true` | Computes Kendall τ per speaker on 60 bin means |
| `global_trend_weighting` | `"equal_language"` | Each language contributes equally to the global trend curve |
