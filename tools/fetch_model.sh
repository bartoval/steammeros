#!/usr/bin/env bash
# Downloads the SCNet-large ONNX model from this repository's releases and checks it.
#
#   tools/fetch_model.sh         # -> models/scnet-large/scnet-large.onnx
#   tools/fetch_model.sh cuda    # also the fast GPU model and its operator library (Linux x86-64)
#
# To build them yourself instead, see tools/export_onnx.py (needs PyTorch and the
# checkpoint), tools/export_cuda_onnx.py and tools/build_cuda_ops.sh.
set -euo pipefail

RELEASE=${STEMS_MODEL_URL:-https://github.com/bartoval/steammeros/releases/download/models-v1}
dir=$(cd "$(dirname "$0")/.." && pwd)/models/scnet-large

# sha256sum on Linux, shasum on macOS.
sha256_ok() {
  if command -v sha256sum > /dev/null; then
    echo "$1  $2" | sha256sum -c "${@:3}" -
  else
    echo "$1  $2" | shasum -a 256 -c "${@:3}" -
  fi
}

# fetch <release asset> <local name> <sha256>
fetch() {
  local file=$dir/$2
  if [[ -f $file ]] && sha256_ok "$3" "$file" --status; then
    echo "already there: $file"
    return
  fi
  mkdir -p "$dir"
  part=$file.part
  curl -fsSL --retry 3 -o "$part" "$RELEASE/$1"
  sha256_ok "$3" "$part"
  mv "$part" "$file"
  part=
  echo "downloaded $file"
}

part=
trap 'if [[ -n $part ]]; then rm -f "$part"; fi' EXIT  # no half-downloaded file left behind

fetch scnet-large.onnx scnet-large.onnx efeb7d69abe78e456ad8fa7609d150cc67f36bff5ce2e78f7743870a56c66432

case ${1:-} in
  "") ;;
  cuda)
    if [[ $(uname -sm) != "Linux x86_64" ]]; then
      echo "the fast GPU model is prebuilt for Linux x86-64 only" >&2
      exit 1
    fi
    fetch scnet-large-cuda.onnx scnet-large-cuda.onnx e5d3c9a07334109a8824a177dcf8b9e96cbf4630c7f7e3aff2a57d03e1041eaa
    fetch libstems_cuda_ops-linux-x64.so libstems_cuda_ops.so f207a76ded8c3ed995d42689e0f31a4ff8972a80791795d8eff2a6dbc199eb57
    ;;
  *)
    echo "usage: $0 [cuda]" >&2
    exit 2
    ;;
esac
