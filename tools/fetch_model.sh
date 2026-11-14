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

if [[ -f $model ]] && echo "$SHA256  $model" | sha256sum -c --status -; then
  echo "already there: $model"
  exit 0
fi

mkdir -p "$(dirname "$model")"
trap 'rm -f "$model.part"' EXIT  # no half-downloaded file left behind
curl -fsSL --retry 3 -o "$model.part" "$URL"
echo "$SHA256  $model.part" | sha256sum -c -
mv "$model.part" "$model"
echo "model in $model"
