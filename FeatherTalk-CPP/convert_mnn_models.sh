#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MNN_BUILD_DIR="${ROOT_DIR}/third_party/MNN/build-macos-arm64"
CONVERTER="${MNN_BUILD_DIR}/MNNConvert"
MODEL_DIR="${ROOT_DIR}/models"

if [[ ! -x "${CONVERTER}" ]]; then
  echo "MNNConvert is missing. Run ./setup_mnn_macos.sh first." >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/feather_hubert.onnx" || ! -f "${MODEL_DIR}/unet_hubert.onnx" ]]; then
  echo "ONNX models are missing. Run tools/export_models.py first." >&2
  exit 1
fi

export DYLD_LIBRARY_PATH="${MNN_BUILD_DIR}:${MNN_BUILD_DIR}/tools/converter${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
"${CONVERTER}" -f ONNX \
  --modelFile "${MODEL_DIR}/feather_hubert.onnx" \
  --MNNModel "${MODEL_DIR}/feather_hubert.mnn" \
  --fp16 \
  --bizCode FeatherTalk
"${CONVERTER}" -f ONNX \
  --modelFile "${MODEL_DIR}/unet_hubert.onnx" \
  --MNNModel "${MODEL_DIR}/unet_hubert.mnn" \
  --fp16 \
  --bizCode FeatherTalk

echo "[convert] ${MODEL_DIR}/feather_hubert.mnn"
echo "[convert] ${MODEL_DIR}/unet_hubert.mnn"
