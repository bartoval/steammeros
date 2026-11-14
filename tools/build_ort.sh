#!/usr/bin/env bash
# Builds ONNX Runtime (CPU) from source as static libraries, in third_party/onnxruntime.
# Our CMake then merges them into libstems.a, so nothing ships as a .so.
#
#   tools/build_ort.sh        # ~30-60 min the first time
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/third_party/onnxruntime
ORT_VERSION=$(sed -n 's/^onnxruntime==//p' "$ROOT/tools/requirements.lock")
ORT_COMMIT=6af1a932e2286d5c58166085d24c68783033edee

if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "v$ORT_VERSION" https://github.com/microsoft/onnxruntime "$SRC"
fi

actual_commit=$(git -C "$SRC" rev-parse HEAD)
if [ "$actual_commit" != "$ORT_COMMIT" ]; then
  printf 'ONNX Runtime revision mismatch: expected %s (v%s), found %s\n' \
    "$ORT_COMMIT" "$ORT_VERSION" "$actual_commit" >&2
  exit 1
fi
if ! git -C "$SRC" diff --quiet HEAD --; then
  printf 'ONNX Runtime has modified tracked files; refusing to build an unpinned runtime.\n' >&2
  exit 1
fi
actual_version=$(tr -d '\r\n' < "$SRC/VERSION_NUMBER")
if [ "$actual_version" != "$ORT_VERSION" ]; then
  printf 'ONNX Runtime version mismatch: requirements pin %s, source reports %s\n' \
    "$ORT_VERSION" "$actual_version" >&2
  exit 1
fi

cd "$SRC"
# --no_telemetry: since 1.31 ORT builds Microsoft's 1DS telemetry SDK (with curl) by default.
#   The library must never touch the network, so it's off.
# --disable_ml_ops: drops the classic-ML operators (trees, SVM...), we only need neural-net ops.
# PIC: the archive ends up inside Go executables, which are position independent.
./build.sh --config Release --parallel --skip_tests --compile_no_warning_as_error \
  --no_telemetry --disable_ml_ops \
  --cmake_extra_defines CMAKE_POSITION_INDEPENDENT_CODE=ON onnxruntime_BUILD_UNIT_TESTS=OFF

# ORT uses re2 (regex) but in a static-only build nothing links it, so it never gets built.
cmake --build build/Linux/Release --target re2 --parallel
