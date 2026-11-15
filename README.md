# steammeros

A Go library that separates music into **drums, bass, other and vocals**.
Uses a C++ core, SCNet-large and ONNX Runtime, inside your program.

Requires **Go 1.26** and a **C++20 compiler** (cgo). Tested on Linux x86-64;
CPU packages also exist for Linux arm64 and Apple Silicon, currently untested.

## Use in Go

```sh
go get github.com/bartoval/steammeros/stems
```

```go
import "github.com/bartoval/steammeros/stems"

opts, err := stems.Prepare(ctx, stems.PrepareOptions{CUDA: true})
if err != nil { panic(err) }
engine, err := stems.NewEngine(opts)
if err != nil { panic(err) }
defer engine.Close()

parts, err := engine.Separate(ctx, audio, 44100, nil)
if err != nil { panic(err) }
```

`Prepare` downloads pinned model/runtime assets once, checks SHA-256 hashes and
reuses the verified user cache. On Linux x86-64, `CUDA: true` also installs the
CUDA 13 and cuDNN 9 runtimes; an NVIDIA driver supporting them is required.
Python, pip and the CUDA toolkit are not needed. Set `CUDA: false` for CPU.

- `Offline: true` prevents downloads. `ArchiveDir` can supply the original archives
  and model files locally; `CacheDir` chooses the installation directory, which can
  be shipped with your app. Keep the included license files when redistributing.
- `NewEngine` and `Separate` never download anything. Prepare during installation
  or explicitly on first use; reuse the engine for subsequent separations.
- `audio` is planar `[][]float32`: mono or stereo at common rates from 8 to 384 kHz.
  Each result has `Name` and `Channels`. Cancel `ctx` to stop; replace `nil` with
  `func(progress float64)` for progress from 0 to 1.
- Concurrent calls are supported; each CPU separation needs roughly 7 GB of RAM.
- For custom model/runtime paths, use `NewEngine(Options{...})` directly.

A runnable [consumer example](examples/consumer/main.go) demonstrates preparation
and inference. An [HTTP example](examples/api/main.go) handles uploads and jobs.

## Command line

From a checkout, with **ffmpeg** installed:

```sh
go run ./cmd/separate -prepare -cuda 0 -out stems song.mp3
```

`-prepare` permits the initial download; omit it for later offline runs. Omit
`-cuda 0` for CPU. Output: `stems/song/{drums,bass,other,vocals}.wav`.
The library itself accepts audio arrays and does not require ffmpeg.

## Fast GPU mode

Add `Fast: true` to `PrepareOptions{CUDA: true}`, or `-fast` to the CLI.
This optional Linux x86-64 mode uses FP16 LSTMs with FP32 arithmetic in cuDNN;
its output differs from the standard FP32 model. The matching operator library
is installed automatically alongside its model and runtimes.

## C/C++ and static builds

The [C API](stems/stems.h) supports separation, progress and cancellation.
To link ONNX Runtime statically on Linux:

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

The standard FP32 model is checked against original PyTorch: SNR ≥50 dB per stem
and ≥40 dB in non-silent 1 s windows; median SDR loss ≤0.1 dB per stem.
See [parity.py](tools/parity.py) and [sdr.py](tools/sdr.py).

## License

[Apache-2.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md).
