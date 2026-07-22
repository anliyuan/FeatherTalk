#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
STB_DIR="${THIRD_PARTY_DIR}/stb"

mkdir -p "${THIRD_PARTY_DIR}" "${STB_DIR}"

if [[ ! -f "${STB_DIR}/stb_image.h" ]]; then
  echo "[setup] Downloading stb image headers..."
  curl --fail --location --retry 3 \
    https://raw.githubusercontent.com/nothings/stb/master/stb_image.h \
    --output "${STB_DIR}/stb_image.h"
fi

if [[ ! -f "${STB_DIR}/stb_image_write.h" ]]; then
  echo "[setup] Downloading stb image writer header..."
  curl --fail --location --retry 3 \
    https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h \
    --output "${STB_DIR}/stb_image_write.h"
fi

echo "[setup] stb image headers are ready under ${STB_DIR}"
