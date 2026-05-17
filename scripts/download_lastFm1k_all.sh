#!/usr/bin/env bash
# Download ALL LastFM-1K artifacts (ratings.tsv) as a single bundle from
# OSF and extract them into the data folder. Equivalent end result to
# download_lastFm1k_dataset.sh, but sourced from the project's own OSF
# storage rather than the upstream mtg.upf.edu archive.
#
# Run from the repository root: `bash scripts/download_lastFm1k_all.sh`.
set -euo pipefail

OSF_URL="https://osf.io/cu4g9/download"
DEST="data/lastFm1k"
ZIP="${DEST}/lastFm1k-artifacts.zip"

mkdir -p "${DEST}"

echo "Downloading LastFM-1K artifacts from ${OSF_URL} ..."
wget --no-verbose --tries=3 -O "${ZIP}" "${OSF_URL}"

echo "Extracting into ${DEST}/ ..."
unzip -o "${ZIP}" -d "${DEST}"
rm -f "${ZIP}"

echo "Done. LastFM-1K artifacts in ${DEST}/:"
ls -1 "${DEST}"
