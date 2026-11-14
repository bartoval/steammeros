#!/usr/bin/env bash
# Downloads the SCNet-large ONNX model from this repository's releases and checks it.
#
#   tools/fetch_model.sh    # -> models/scnet-large/scnet-large.onnx
#
# To build it yourself instead, see tools/export_onnx.py (needs PyTorch and the checkpoint).
set -euo pipefail

SHA256=efeb7d69abe78e456ad8fa7609d150cc67f36bff5ce2e78f7743870a56c66432
URL=${STEMS_MODEL_URL:-https://github.com/bartoval/steammeros/releases/download/models-v1/scnet-large.onnx}

root=$(cd "$(dirname "$0")/.." && pwd)
model=$root/models/scnet-large/scnet-large.onnx

# sha256sum on Linux, shasum on macOS.
sha256_ok() {
  if command -v sha256sum > /dev/null; then
    echo "$SHA256  $1" | sha256sum -c "${@:2}" -
  else
    echo "$SHA256  $1" | shasum -a 256 -c "${@:2}" -
  fi
}

if [[ -f $model ]] && sha256_ok "$model" --status; then
  echo "already there: $model"
  exit 0
fi

mkdir -p "$(dirname "$model")"
trap 'rm -f "$model.part"' EXIT  # no half-downloaded file left behind
curl -fsSL --retry 3 -o "$model.part" "$URL"
sha256_ok "$model.part"
mv "$model.part" "$model"
echo "model in $model"
