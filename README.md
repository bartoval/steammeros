# steammeros

A Go library that separates music into **drums, bass, other and vocals**.
Uses a C++ core, SCNet-large and ONNX Runtime. Runs inside your program, without
Python at runtime.

Early release, tested on Linux x86-64. macOS on Apple Silicon is untested.

## Quick start

Requires **Go 1.26**, a **C++20 compiler**, **ONNX Runtime 1.31+** and the model.
The command-line tool also needs **ffmpeg** to decode audio.

```sh
git clone https://github.com/bartoval/steammeros && cd steammeros
tools/fetch_ort.sh
export STEMS_ORT_LIBRARY=...       # use the path printed by the script
tools/fetch_model.sh
go run ./cmd/separate -out stems song.mp3
```

Output: `stems/song/{drums,bass,other,vocals}.wav`.

## Use in Go

```sh
go get github.com/bartoval/steammeros/stems
```

```go
import "github.com/bartoval/steammeros/stems"

engine, err := stems.NewEngine(stems.Options{
    ModelPath: "models/scnet-large/scnet-large.onnx",
})
if err != nil { panic(err) }
defer engine.Close()

parts, err := engine.Separate(ctx, audio, 44100, nil)
if err != nil { panic(err) }
```

- `audio` is planar `[][]float32`: one slice per channel, mono or stereo, at common
  sample rates from 8 to 384 kHz. Each result has `Name` and `Channels`.
- Reuse the engine across calls. `Separate` blocks; cancel `ctx` to stop it.
  Replace `nil` with `func(progress float64)` for progress from 0 to 1.
- Set `Options.ORTLibrary` or `STEMS_ORT_LIBRARY` to locate ONNX Runtime.
- Concurrent calls are supported, but each CPU separation needs roughly 7 GB of RAM.

An [HTTP server example](examples/api/main.go) supports uploads, progress,
cancellation and stem downloads: `go run ./examples/api`.

## NVIDIA GPU

Requires **CUDA 13**, **cuDNN 9** and the GPU build of ONNX Runtime:

```sh
tools/fetch_ort.sh gpu
export STEMS_ORT_LIBRARY=...       # use the GPU runtime path printed by the script
go run ./cmd/separate -cuda 0 -out stems song.mp3
```

In Go, set `CUDA: true` in the engine options.

### Fast mode (Linux)

A second model runs the LSTMs in FP16 through a small cuDNN library, about **1.6x
faster**: a 253 s track takes **15.4 s instead of 25.2 s** on an RTX 4060 Laptop GPU,
with the same VRAM (~6.3 GB).

```sh
tools/fetch_model.sh cuda          # scnet-large-cuda.onnx and libstems_cuda_ops.so
go run ./cmd/separate -cuda 0 -model models/scnet-large/scnet-large-cuda.onnx \
  -cuda-ops models/scnet-large/libstems_cuda_ops.so -out stems song.mp3
```

In Go, use that `ModelPath` and set `CUDAOpsLibrary`. Separation quality (SDR) matches
the standard model; outputs differ by a residual typically 70 dB below the signal, so
this mode isn't held to the parity checks below.
To build both files yourself: [export_cuda_onnx.py](tools/export_cuda_onnx.py),
[build_cuda_ops.sh](tools/build_cuda_ops.sh).

## C/C++ and static builds

The [C API](stems/stems.h) supports separation, progress and cancellation.
To build with ONNX Runtime linked statically on Linux:

```sh
tools/build_ort.sh
cmake -S . -B build -G Ninja && cmake --build build
PKG_CONFIG_PATH=$PWD/build/pkgconfig go build -tags stems_static ./...
```

After rebuilding the archives, add `-a` to the Go build command.

## Model and validation

SCNet-large by starrytong, under MIT, including its weights
([author's confirmation](https://github.com/starrytong/SCNet/issues/35#issuecomment-4999873539)).
The checkpoint is recorded in [models/manifest.json](models/manifest.json).

Output is checked against the original PyTorch model: SNR ≥50 dB per stem and ≥40 dB
in non-silent 1 s windows; median SDR loss ≤0.1 dB per stem. The scripts are
[parity.py](tools/parity.py) and [sdr.py](tools/sdr.py); Python dependencies are pinned
in [requirements.lock](tools/requirements.lock).

## License

[Apache-2.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md).
