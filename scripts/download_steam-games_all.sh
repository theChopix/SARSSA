#!/usr/bin/env bash
set -euo pipefail

OSF_URL="https://osf.io/8j64a/download"
DEST="data/steam-games"
ZIP="${DEST}/steam-games-artifacts.zip"

mkdir -p "${DEST}"

echo "Downloading Steam Games artifacts from ${OSF_URL} ..."
wget --no-verbose --tries=3 -O "${ZIP}" "${OSF_URL}"

echo "Extracting into ${DEST}/ ..."
unzip -o "${ZIP}" -d "${DEST}"
rm -f "${ZIP}"

echo "Done. Steam Games artifacts in ${DEST}/:"
ls -1 "${DEST}"
