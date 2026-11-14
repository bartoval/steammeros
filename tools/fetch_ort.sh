#!/usr/bin/env bash
# Downloads an official prebuilt ONNX Runtime release and checks it. Used by the dev build
# (quick setup, also on macOS) and by the GPU build; both link it dynamically.
#
#   tools/fetch_ort.sh        # CPU, for this machine -> third_party/onnxruntime-prebuilt
#   tools/fetch_ort.sh gpu    # Linux x64, CUDA 13    -> third_party/onnxruntime-gpu
set -euo pipefail

VERSION=1.31.1  # same as tools/requirements.lock; the checksums below are for this version

case "${1:-cpu}-$(uname -s)-$(uname -m)" in
  cpu-Linux-x86_64)
    name=onnxruntime-linux-x64-$VERSION
    sha=c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9
    folder=onnxruntime-prebuilt ;;
  cpu-Linux-aarch64)
    name=onnxruntime-linux-aarch64-$VERSION
    sha=5317505e2da99be65f8e2775ba3c2afd6d2961fd78aba1bd8f56159f48820c78
    folder=onnxruntime-prebuilt ;;
  cpu-Darwin-arm64)
    name=onnxruntime-osx-arm64-$VERSION
    sha=2e1344a1926ee50eb25b38153cb6992c3567515aef0e677a1a520a1798896d6f
    folder=onnxruntime-prebuilt ;;
  gpu-Linux-x86_64)
    name=onnxruntime-linux-x64-gpu_cuda13-$VERSION
    sha=b2146862e691f7e338f5aff9c7a9a5567c2ca57cf01442a1adb3a13bf62d9095
    folder=onnxruntime-gpu ;;
  *)
    echo "no prebuilt ONNX Runtime for '${1:-cpu}' on $(uname -s) $(uname -m)" >&2
    exit 1 ;;
esac

root=$(cd "$(dirname "$0")/.." && pwd)
dest=$root/third_party/$folder
archive=$root/third_party/$name.tgz

# Tells how to point the Go module at the library (see Options.ORTLibrary).
print_usage() {
  local lib=$dest/lib/libonnxruntime.so
  [[ $(uname -s) == Darwin ]] && lib=$dest/lib/libonnxruntime.dylib
  echo "export STEMS_ORT_LIBRARY=$lib"
}

if [[ -f $dest/VERSION_NUMBER && $(cat "$dest/VERSION_NUMBER") == "$VERSION" ]]; then
  echo "already there: $dest"
  print_usage
  exit 0
fi

# sha256sum on Linux, shasum on macOS.
check_sha256() {
  if command -v sha256sum > /dev/null; then
    echo "$2  $1" | sha256sum -c -
  else
    echo "$2  $1" | shasum -a 256 -c -
  fi
}

# A go.mod makes Go's ./... skip this folder (ONNX Runtime's sources contain Go files).
mkdir -p "$root/third_party" && echo "module thirdparty" > "$root/third_party/go.mod"
trap 'rm -f "$archive"' EXIT
curl -fsSL --retry 3 -o "$archive" "https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/$name.tgz"
check_sha256 "$archive" "$sha"

rm -rf "$dest"
mkdir -p "$dest"
tar -xzf "$archive" -C "$dest" --strip-components=1
echo "ONNX Runtime $VERSION ($name) in $dest"
print_usage
