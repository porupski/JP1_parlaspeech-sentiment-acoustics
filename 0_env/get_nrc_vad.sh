#!/usr/bin/env bash
# Downloads NRC-VAD-Lexicon v2.1 and places the files in data/lexicons/.
# Run from the repository root.
set -e

DEST_DIR="data/lexicons"
ZIP_NAME="NRC-VAD-Lexicon-v2.1.zip"
EXTRACT_DIR="${DEST_DIR}/NRC-VAD-Lexicon-v2.1"
URL="https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"

mkdir -p "${DEST_DIR}"

if [ -d "${EXTRACT_DIR}" ]; then
    echo "NRC VAD Lexicon already present at ${EXTRACT_DIR}"
    exit 0
fi

echo "Downloading NRC-VAD-Lexicon v2.1 ..."
curl -L -o "${DEST_DIR}/${ZIP_NAME}" "${URL}"

echo "Unzipping ..."
unzip -q "${DEST_DIR}/${ZIP_NAME}" -d "${DEST_DIR}"

rm "${DEST_DIR}/${ZIP_NAME}"

echo ""
echo "Lexicon files placed in: ${EXTRACT_DIR}"
echo "Contents:"
ls "${EXTRACT_DIR}"
echo ""
echo "Check config.json → vad.nrc_vad_path points to the correct .txt file above."
