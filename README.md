# steammeros

In-process music source separation: splits a song into drums, bass, other and vocals,
inside your program. A C++20 core with a C API and a Go package (cgo), linked statically:
no server, no Python, nothing downloaded at run time. Model: SCNet-large, matching the
original PyTorch model (see [Validation](#validation)).

Early release: Linux x86-64 only, and the API may still change.

## Quick start

Needs git, CMake ≥ 3.28, Ninja, GCC with C++20, Python 3 (ONNX Runtime's build script)
and Go 1.26; the example command also needs ffmpeg.

```sh
git clone https://github.com/bartoval/steammeros && cd steammeros
tools/build_ort.sh                  # builds ONNX Runtime 1.31.1 from source, ~30-60 min once
cmake -S . -B build -G Ninja && cmake --build build
tools/fetch_model.sh                # 175 MB ONNX model from the releases, checksum verified
export PKG_CONFIG_PATH=$PWD/build/pkgconfig
cd go && go run ./cmd/separate -model ../models/scnet-large/scnet-large.onnx -out ../stems song.mp3
```

That writes `stems/song/{drums,bass,other,vocals}.wav`.

## Go

```go
import stems "github.com/bartoval/steammeros/go"

engine, err := stems.NewEngine(stems.Options{ModelPath: "models/scnet-large/scnet-large.onnx"})
if err != nil { ... }
defer engine.Close()

// audio: planar float32, audio[channel][sample], mono or stereo, 8 to 384 kHz.
parts, err := engine.Separate(ctx, audio, 44100, func(done float64) { /* 0..1 */ })
for _, part := range parts {
    // part.Name: "drums", "bass", "other" or "vocals"; part.Channels like audio
}
```

- **Blocking call:** `Separate` blocks until done. Cancelling `ctx` stops it within a
  moment, even in the middle of the model, and returns `ctx.Err()`.
- **Concurrency:** an `Engine` is safe to use from several goroutines.
- **Linking:** your module links `libstems.a` through pkg-config, so build it as above and
  set `PKG_CONFIG_PATH` before `go get github.com/bartoval/steammeros/go`.
- **Rebuilding:** after rebuilding the archive, pass `-a` to `go build`/`go test`. Go
  caches cgo builds and doesn't notice changes inside `libstems.a`.

The C API (`include/stems.h`) offers the same: `stems_separate` plus an optional
`stems_job` to read progress and cancel from another thread.

## GPU (NVIDIA)

```sh
tools/fetch_ort_gpu.sh              # official ONNX Runtime 1.31.1 GPU release (CUDA 13)
cmake -S . -B build-cuda -G Ninja -DSTEMS_CUDA=ON && cmake --build build-cuda
export PKG_CONFIG_PATH=$PWD/build-cuda/pkgconfig
```

Then set `CUDA: true` in the Go options (`-cuda 0` in `cmd/separate`). This build links
ONNX Runtime dynamically and needs CUDA 13 and cuDNN 9 at run time (`libcudart`,
`libcublas`, `libcurand`, `libcufft`, `libcudnn`). Precision stays FP32 with TF32 off,
so GPU output matches the CPU and the original model. A 253 s track takes ~30 s on an
RTX 4060 Laptop GPU and uses ~6.3 GB of VRAM.

## How it works

- **Same pipeline as the original:** audio is resampled to 44.1 kHz and back (a port of
  `julius.resample_frac`), normalized, cut into 20 s chunks with 50% overlap, and
  blended with triangular weights.
- **Native spectrograms:** the STFT/iSTFT run in C++ (pocketfft); only the network itself
  runs in ONNX Runtime.
- **Pipelined chunks:** with more than one thread, the next chunk's spectrum and the
  previous chunk's reconstruction overlap the model run. `threads=1` keeps everything
  sequential.
- **Pinned runtime:** ONNX Runtime is pinned to 1.31.1 (`tools/requirements.lock`).
  `tools/build_ort.sh` checks out and verifies commit
  `6af1a932e2286d5c58166085d24c68783033edee`, and the core checks the linked runtime
  version before creating an engine.

## Model

SCNet-large by starrytong (commit `5d95bf9`), four stems, 44.1 kHz stereo. MIT license,
weights included ([author's confirmation](https://github.com/starrytong/SCNet/issues/35#issuecomment-4999873539)).
`tools/fetch_model.sh` gets the ONNX export from the
[models-v1 release](https://github.com/bartoval/steammeros/releases/tag/models-v1).
To export it yourself, run `tools/export_onnx.py` with the checkpoint listed in
`models/manifest.json`.

## Validation

### Criteria

- **Parity:** the output must match the original PyTorch FP32 model run with
  `shifts=0` (the original uses a random shift). The SNR of the difference must be at
  least 50 dB per stem, and at least 40 dB in the worst 1 s window. Channels, length and
  stem order must be exact.
- **Quality:** SDR on the MUSDB18-HQ test set
  ([Zenodo 3338373](https://zenodo.org/records/3338373)). The median drop must be at
  most 0.1 dB per stem.
- **CPU build:** links system libraries only (libc, libm, libstdc++, libgcc_s).

### Results

CPU, every stem passes.

| Set | Stems | Min SNR | Worst 1 s window | Worst SDR change |
|-|-:|-:|-:|-:|
| 12 MUSDB excerpts, 25 s | 48 | 91.6 dB | 85.6 dB | -0.000015 dB |
| 3 full MUSDB tracks, 76-430 s | 12 | 98.1 dB | 84.5 dB | -0.000001 dB |
| 48 kHz, 22.05 kHz, mono inputs | 20 | 112.7 dB | 89.4 dB | (no true stems) |

On a Ryzen 7 7840HS with 8 threads, CPU separation runs at about 0.75-1.0x real time.
It peaks at 6.9-7.7 GB of RAM on 4-7 minute tracks.

### Reproducing


Reference stems come from `tools/run_reference.py` (original model, PyTorch), and the
excerpts from `tools/make_excerpts.py`. The Python tools are pinned in
`tools/requirements.lock`.

```sh
build/separate -model models/scnet-large/scnet-large.onnx -out data/results/native data/excerpts/*/mixture.wav
.venv/bin/python -I tools/parity.py data/results/native data/results/excerpts-deterministic
.venv/bin/python -I tools/sdr.py data/results/native data/excerpts --baseline data/results/excerpts-deterministic
```

## License

Apache-2.0 ([LICENSE](LICENSE)). Third-party components in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
