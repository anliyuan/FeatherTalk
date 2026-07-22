#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
MNN_DIR="${THIRD_PARTY_DIR}/MNN"
MNN_BUILD_DIR="${MNN_DIR}/build-macos-arm64"
CACHE_DIR="${THIRD_PARTY_DIR}/.downloads"
MNN_ARCHIVE="${CACHE_DIR}/MNN-master.tar.gz"
MNN_URL="https://codeload.github.com/alibaba/MNN/tar.gz/refs/heads/master"
SDK_DIR="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
CLANG="/Library/Developer/CommandLineTools/usr/bin/clang"
CLANGXX="/Library/Developer/CommandLineTools/usr/bin/clang++"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "${CLANG}" || ! -x "${CLANGXX}" || ! -d "${SDK_DIR}" ]]; then
  echo "Command Line Tools are required. Install them with: xcode-select --install" >&2
  exit 1
fi

if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null; then
  echo "[setup] Installing cmake and ninja with ${PYTHON_BIN}..."
  "${PYTHON_BIN}" -m pip install cmake ninja
  export PATH="$(${PYTHON_BIN} -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):${PATH}"
fi

CMAKE_BIN="$(command -v cmake)"
NINJA_BIN="$(command -v ninja)"
mkdir -p "${THIRD_PARTY_DIR}" "${CACHE_DIR}"

if [[ ! -f "${MNN_DIR}/CMakeLists.txt" ]]; then
  if [[ -e "${MNN_DIR}" ]]; then
    echo "${MNN_DIR} exists but is not a valid MNN source tree. Move it aside and retry." >&2
    exit 1
  fi
  echo "[setup] Downloading MNN source..."
  curl --continue-at - --fail --location --retry 3 "${MNN_URL}" --output "${MNN_ARCHIVE}"
  temp_dir="$(mktemp -d)"
  trap 'rm -rf "${temp_dir}"' EXIT
  tar -xzf "${MNN_ARCHIVE}" --strip-components=1 -C "${temp_dir}"
  mv "${temp_dir}" "${MNN_DIR}"
  trap - EXIT
fi

mkdir -p "${MNN_BUILD_DIR}"
"${CMAKE_BIN}" -S "${MNN_DIR}" -B "${MNN_BUILD_DIR}" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="${NINJA_BIN}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CLANG}" \
  -DCMAKE_CXX_COMPILER="${CLANGXX}" \
  -DCMAKE_OSX_SYSROOT="${SDK_DIR}" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_SEP_BUILD=OFF \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_METAL=ON \
  -DMNN_OPENCL=ON \
  -DMNN_BUILD_DEMO=OFF \
  -DMNN_BUILD_TOOLS=OFF \
  -DMNN_AAPL_FMWK=OFF
"${CMAKE_BIN}" --build "${MNN_BUILD_DIR}" --target MNN MNNConvert --parallel 6

echo "[setup] MNN runtime: ${MNN_BUILD_DIR}/libMNN.dylib"
echo "[setup] MNN converter: ${MNN_BUILD_DIR}/MNNConvert"
