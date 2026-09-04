#!/usr/bin/env bash
# Placeholder: downloads ParlaSpeech v4.0 JSONL files to data/parlaspeech/.
# Fill in the URL or path when the dataset is publicly hosted.
# Run from any directory: bash 0_env/get_parlaspeech_v4.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEST_DIR="${REPO_ROOT}/data/parlaspeech"

mkdir -p "${DEST_DIR}"

echo "ERROR: ParlaSpeech v4.0 is not yet publicly hosted."
echo "Place the .patched.jsonl files in ${DEST_DIR} manually,"
echo "then update config.json → paths.data_root to point to that directory."
exit 1

# --- Template for when hosting is available ---
# LANGS=(HR CZ PL RS SI)
# BASE_URL="https://example.com/parlaspeech/v4/"
# for LANG in "${LANGS[@]}"; do
#     FILE="ParlaSpeech-${LANG}.v4.0.patched.jsonl"
#     echo "Downloading ${FILE} ..."
#     curl -L -o "${DEST_DIR}/${FILE}" "${BASE_URL}${FILE}"
# done
# echo "Done. Update config.json → paths.data_root to ${DEST_DIR}"
