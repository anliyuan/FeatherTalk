#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MNN_DIR="${ROOT_DIR}/third_party/MNN"
MNN_BUILD_DIR="${MNN_DIR}/build-macos-arm64"
SDK_DIR="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
CLANG="/Library/Developer/CommandLineTools/usr/bin/clang++"
OUT_DIR="${ROOT_DIR}/bin"

if [[ ! -x "${CLANG}" || ! -d "${SDK_DIR}" ]]; then
  echo "Command Line Tools are required. Install them with: xcode-select --install" >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/third_party/stb/stb_image.h" ]]; then
  bash "${ROOT_DIR}/setup_macos.sh"
fi
if [[ ! -f "${MNN_BUILD_DIR}/libMNN.dylib" ]]; then
  bash "${ROOT_DIR}/setup_mnn_macos.sh"
fi

mkdir -p "${OUT_DIR}"
"${CLANG}" \
  -std=c++17 -O3 -DNDEBUG \
  -isysroot "${SDK_DIR}" \
  -isystem "${SDK_DIR}/usr/include/c++/v1" \
  -I"${MNN_DIR}/include" \
  -I"${ROOT_DIR}/third_party/stb" \
  "${ROOT_DIR}/src/main.cc" \
  -L"${MNN_BUILD_DIR}" -lMNN \
  -Wl,-rpath,@executable_path/../third_party/MNN/build-macos-arm64 \
  -o "${OUT_DIR}/feathertalk_mnn"

"${CLANG}" \
  -std=c++17 -O3 -DNDEBUG \
  -isysroot "${SDK_DIR}" \
  -isystem "${SDK_DIR}/usr/include/c++/v1" \
  -I"${MNN_DIR}/include" \
  "${ROOT_DIR}/src/benchmark_mnn.cc" \
  -L"${MNN_BUILD_DIR}" -lMNN \
  -Wl,-rpath,@executable_path/../third_party/MNN/build-macos-arm64 \
  -o "${OUT_DIR}/benchmark_mnn"

echo "[build] ${OUT_DIR}/feathertalk_mnn"
echo "[build] ${OUT_DIR}/benchmark_mnn"
