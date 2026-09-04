#!/usr/bin/env bash
# ============================================================
# Setup script for parlaspeech-sentiment-acoustics env
# ============================================================
set -e

ENV_NAME="jp1_ps_sent"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating mamba env: $ENV_NAME"
mamba env create -f "$SCRIPT_DIR/environment.yml" -n "$ENV_NAME"

echo "Activating env ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Converting editorial notebook ..."
cd "$(dirname "$SCRIPT_DIR")/4_outputs"
jupytext --to notebook 42_plots_editorial.py 2>/dev/null || true

echo ""
echo "Done. To activate:"
echo "  mamba activate $ENV_NAME"
echo ""
echo "Then fill in audio paths in config.json and run:"
echo "  python run_all.py --skip praat opensmile  # speech rate only first"
echo "  python run_all.py                          # full pipeline"
