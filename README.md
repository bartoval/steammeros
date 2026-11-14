# stems

In-process music source separation: a C++20 core with a C API, a Go package via cgo,
everything linked statically. Model: SCNet-large.

Status: in development, not ready for use yet. Native CPU separation matches the
original PyTorch model (see Validation below).

## Build

```sh
tools/build_ort.sh
cmake -S . -B build -G Ninja && cmake --build build
export PKG_CONFIG_PATH=$PWD/build/pkgconfig
(cd go && go test ./...)
```

`go get` alone isn't enough: the native archive has to be built first.

ONNX Runtime is pinned to **1.31.1** (`tools/requirements.lock`), for the native build
and for the Python tools. `tools/build_ort.sh` checks out commit
`6af1a932e2286d5c58166085d24c68783033edee` (tag `v1.31.1`), verifies it even when the
checkout already exists, and refuses local changes to tracked files. The export checks
the Python ONNX Runtime version; the core checks the linked runtime version before
creating an engine.

## API

Synchronous for now: float32 planar PCM, mono or stereo, 8 to 384 kHz. As in the
original, audio goes to 44.1 kHz for the model and back to the input rate (a port of
`julius.resample_frac`). Verified on Linux x86-64, CPU.

With more than one thread, the pipeline prepares the next chunk's spectrum and
reconstructs the previous one while the model runs on the current chunk. Up to four
workers run the iSTFTs on independent channels; the overlap-add order doesn't change.
`threads` sets the inference threads, on top of those workers; `threads=1` keeps
everything sequential.

## Model

SCNet-large by starrytong (commit `5d95bf9`). MIT license, weights included
([author's confirmation](https://github.com/starrytong/SCNet/issues/35#issuecomment-4999873539)).
Four stems (drums, bass, other, vocals), 44.1 kHz stereo. Sources and checksums in
`models/manifest.json`; the weights aren't in Git.

## Validation

- Parity with the original PyTorch FP32 model (with `shifts=0`; the original uses a
  random shift): SNR of the difference ≥ 50 dB per stem; exact channels, length and
  stem order.
- Quality: SDR on the MUSDB18-HQ test set; median drop at most 0.1 dB per stem.
  Tracks from [Zenodo 3338373](https://zenodo.org/records/3338373), extracted to
  `data/musdb18hq/test/`.
- CPU build: system libraries only (libc, libm, libstdc++, libgcc_s).

Reference stems come from the original model (`tools/run_reference.py`); the corpus is
12 excerpts picked by `tools/make_excerpts.py`, plus full tracks and 48/22.05 kHz inputs.

```sh
build/separate -model models/scnet-large/scnet-large.onnx -out data/results/native data/excerpts/*/mixture.wav
.venv/bin/python -I tools/parity.py data/results/native data/results/excerpts-deterministic
.venv/bin/python -I tools/sdr.py data/results/native data/excerpts --baseline data/results/excerpts-deterministic
```

## License

Apache-2.0 ([LICENSE](LICENSE)). Third-party components in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
