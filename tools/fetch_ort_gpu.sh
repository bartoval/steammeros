#!/usr/bin/env bash
# Downloads the official ONNX Runtime GPU build (CUDA 13) for the STEMS_CUDA build.
#
#   tools/fetch_ort_gpu.sh    # -> third_party/onnxruntime-gpu/{include,lib}
#
# Unlike the CPU build, this one links ONNX Runtime dynamically: its CUDA provider is a
# shared library by design. Running it needs CUDA 13 and cuDNN 9 on the system.
set -euo pipefail

VERSION=1.31.1
NAME=onnxruntime-linux-x64-gpu_cuda13-$VERSION
SHA256=b2146862e691f7e338f5aff9c7a9a5567c2ca57cf01442a1adb3a13bf62d9095
URL=https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/$NAME.tgz

root=$(cd "$(dirname "$0")/.." && pwd)
dest=$root/third_party/onnxruntime-gpu
archive=$root/third_party/$NAME.tgz

if [[ -f $dest/VERSION_NUMBER && $(cat "$dest/VERSION_NUMBER") == "$VERSION" ]]; then
  echo "already there: $dest"
  exit 0
fi

mkdir -p "$root/third_party"
curl -fsSL --retry 3 -o "$archive" "$URL"
echo "$SHA256  $archive" | sha256sum -c -

rm -rf "$dest"
mkdir -p "$dest"
tar -xzf "$archive" -C "$dest" --strip-components=1
rm "$archive"
echo "ONNX Runtime $VERSION (CUDA 13) in $dest"
