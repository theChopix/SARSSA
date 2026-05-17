#!/usr/bin/env bash
# Download ALL MovieLens artifacts (ratings.csv, tags.csv, metadata.json,
# descriptions.json) as a single bundle from OSF and extract them into
# the data folder. Unlike download_movieLens_dataset.sh (which fetches
# only the raw dataset from GroupLens), this also provides the curated
# metadata.json / descriptions.json needed for the full pipeline + UI.
#
# Run from the repository root: `bash scripts/download_movieLens_all.sh`.
set -euo pipefail

OSF_URL="https://osf.io/h82uq/download"
DEST="data/movieLens"
ZIP="${DEST}/movieLens-artifacts.zip"

mkdir -p "${DEST}"

echo "Downloading MovieLens artifacts from ${OSF_URL} ..."
wget --no-verbose --tries=3 -O "${ZIP}" "${OSF_URL}"

echo "Extracting into ${DEST}/ ..."
unzip -o "${ZIP}" -d "${DEST}"
rm -f "${ZIP}"

echo "Done. MovieLens artifacts in ${DEST}/:"
ls -1 "${DEST}"
