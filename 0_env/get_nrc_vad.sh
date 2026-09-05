#!/usr/bin/env bash
# Downloads NRC-VAD-Lexicon v1 (multilingual) to data/lexicons/ inside the repository.
# v1 includes Google Translate translations for 100+ languages including HR, CZ, PL, RS, SI.
# v2.1 (English-only) is NOT used for this pipeline.
# Run from any directory: bash 0_env/get_nrc_vad.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEST_DIR="${REPO_ROOT}/data/lexicons"
EXTRACT_DIR="${DEST_DIR}/NRC-VAD-Lexicon-v1"
ZIP_NAME="NRC-VAD-Lexicon-v1.zip"
URL="https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip"

mkdir -p "${DEST_DIR}"

if [ -d "${EXTRACT_DIR}" ]; then
    echo "NRC VAD Lexicon v1 already present at ${EXTRACT_DIR}"
    exit 0
fi

if [ ! -f "${DEST_DIR}/${ZIP_NAME}" ]; then
    echo "Downloading NRC-VAD-Lexicon v1 (multilingual) to ${DEST_DIR} ..."
    curl -L -o "${DEST_DIR}/${ZIP_NAME}" "${URL}"
fi

echo "Extracting ..."
if command -v 7z >/dev/null 2>&1; then
    7z x "${DEST_DIR}/${ZIP_NAME}" -o"${EXTRACT_DIR}" -y > /dev/null
else
    python3 -c "import zipfile; zipfile.ZipFile('${DEST_DIR}/${ZIP_NAME}').extractall('${EXTRACT_DIR}')"
fi
rm "${DEST_DIR}/${ZIP_NAME}"

echo ""
echo "Files placed in: ${EXTRACT_DIR}"
ls "${EXTRACT_DIR}"
echo ""
echo "After extraction, verify per-language files with:"
echo "  ls ${EXTRACT_DIR}/NRC-VAD-Lexicon/OneFilePerLanguage/ | head"
echo "config.json → vad.nrc_vad_dir already points to the correct nested path."
