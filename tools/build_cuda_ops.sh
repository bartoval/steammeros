#!/usr/bin/env bash
# Builds the CUDA operator library for the fast GPU model (Linux). Prebuilt copies come
# with `tools/fetch_model.sh cuda`; build it yourself only to change it.
#
#   CUDA_ROOT=/usr/local/cuda CUDNN_ROOT=/usr tools/build_cuda_ops.sh [output]
#
# CUDA_ROOT and CUDNN_ROOT hold include/ and lib64/ (or lib/). The NVIDIA pip wheels lack
# crt/host_defines.h: add its directory with CXXFLAGS=-I...
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
: "${CUDA_ROOT:?set CUDA_ROOT to a CUDA 13 installation}"
: "${CUDNN_ROOT:?set CUDNN_ROOT to a cuDNN 9 installation}"
output=${1:-$root/build/cuda/libstems_cuda_ops.so}

libdir() { if [[ -d $1/lib64 ]]; then echo "$1/lib64"; else echo "$1/lib"; fi; }

mkdir -p "$(dirname "$output")"
"${CXX:-g++}" -std=c++20 -O2 -shared -fPIC -fvisibility=hidden ${CXXFLAGS:-} \
  -I"$root/stems/external/onnxruntime" -I"$CUDA_ROOT/include" -I"$CUDNN_ROOT/include" \
  "$root/stems/cuda/lstm.cpp" -o "$output" \
  -L"$(libdir "$CUDA_ROOT")" -L"$(libdir "$CUDNN_ROOT")" -l:libcudart.so.13 -l:libcudnn.so.9
echo "built $output"
