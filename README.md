# steammeros

In-process music source separation for Go: splits a song into drums, bass, other and
vocals, inside your program. No server, no Python, nothing downloaded at run time. The
core is C++ (compiled by cgo), the model is SCNet-large, and the output matches the
original PyTorch model (see [Validation](#validation)).

Early release: tested on Linux x86-64; macOS on Apple Silicon should work but is
untested. The API may still change.

## Quick start

```sh
go get github.com/bartoval/steammeros/stems
```

That compiles everything; you need Go 1.26 and a C++20 compiler (GCC or Clang). At run
time two files are needed:

- **ONNX Runtime 1.31 or newer.** It runs the network. Install it with
  `brew install onnxruntime` on macOS, or use `tools/fetch_ort.sh`, which downloads the
  official release and prints the `STEMS_ORT_LIBRARY` line to set. Without a path, the
  usual system locations are searched.
- **The model.** `tools/fetch_model.sh` downloads it (175 MB) and verifies its checksum.

To try it from a clone (needs ffmpeg to decode the input):

```sh
git clone https://github.com/bartoval/steammeros && cd steammeros
tools/fetch_ort.sh                 # prints: export STEMS_ORT_LIBRARY=...
export STEMS_ORT_LIBRARY=...       # the line it printed
tools/fetch_model.sh
go run ./cmd/separate -out stems song.mp3   # -> stems/song/{drums,bass,other,vocals}.wav
```

### As an HTTP API

[`examples/api`](examples/api/main.go) is a small server around the package. You upload a
song, follow the job, and download the stems. Uploads are decoded with ffmpeg, and
separations queue and run one at a time.

```sh
go run ./examples/api                                  # with STEMS_ORT_LIBRARY set as above
curl --data-binary @song.mp3 localhost:8080/jobs       # -> {"id":"1","status":"queued",...}
curl localhost:8080/jobs/1                             # -> {"status":"running","progress":0.4}
curl -o vocals.wav localhost:8080/jobs/1/stems/vocals  # once "done"
curl -X DELETE localhost:8080/jobs/1                   # cancels it if running, deletes files
```

## Go

```go
import "github.com/bartoval/steammeros/stems"

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
- **ONNX Runtime location:** `Options.ORTLibrary` sets its path. When empty,
  `$STEMS_ORT_LIBRARY` is used, then the usual system locations.

In a web server, create one engine at startup and pass the request's context, so a
closed connection cancels the work. Each separation needs about 7 GB of RAM and roughly
real time on CPU, so limit how many run at once. For long tracks, it's better to answer
with a job id than to keep the request open.

## GPU (NVIDIA)

```sh
tools/fetch_ort.sh gpu             # official ONNX Runtime GPU release (CUDA 13)
export STEMS_ORT_LIBRARY=...       # the line it printed
```

Then set `CUDA: true` in the options (`-cuda 0` in `cmd/separate`). CUDA 13 and cuDNN 9
must be installed (`libcudart`, `libcublas`, `libcurand`, `libcufft`, `libcudnn`).
Precision stays FP32 with TF32 off, so GPU output matches the CPU and the original
model. A 253 s track takes ~30 s on an RTX 4060 Laptop GPU, with ~6.3 GB of VRAM.

## Static build

For one self-contained binary that needs only system libraries (Linux), ONNX Runtime can
be built from source and linked in:

```sh
tools/build_ort.sh                 # ONNX Runtime 1.31.1 from source, ~30-60 min once
cmake -S . -B build -G Ninja && cmake --build build
PKG_CONFIG_PATH=$PWD/build/pkgconfig go build -tags stems_static ./...
```

After rebuilding the archive, pass `-a` to `go build`/`go test`. Go doesn't notice
changes inside libraries it links through pkg-config. The same CMake build also produces
`libstems.a` with `pkgconfig/stems.pc`, for C and C++ programs using the C API in
`stems/stems.h`. That API is the same as in Go: `stems_separate` plus an optional `stems_job`
to read progress and cancel from another thread.

## How it works

- **Same pipeline as the original:** audio is resampled to 44.1 kHz and back (a port of
  `julius.resample_frac`), normalized, cut into 20 s chunks with 50% overlap, and
  blended with triangular weights.
- **Native spectrograms:** the STFT/iSTFT run in C++ (pocketfft); only the network itself
  runs in ONNX Runtime.
- **Pipelined chunks:** with more than one thread, the next chunk's spectrum and the
  previous chunk's reconstruction overlap the model run. `Threads: 1` keeps everything
  sequential.
- **Pinned runtime for validation:** results were validated with ONNX Runtime 1.31.1
  (`tools/requirements.lock`). `tools/build_ort.sh` checks out and verifies commit
  `6af1a932e2286d5c58166085d24c68783033edee`.

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

### Results

CPU, every stem passes.

| Set | Stems | Min SNR | Worst 1 s window | Worst SDR change |
| --- | ---: | ---: | ---: | ---: |
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
go run ./cmd/separate -out data/results/native data/excerpts/*/mixture.wav
.venv/bin/python -I tools/parity.py data/results/native data/results/excerpts-deterministic
.venv/bin/python -I tools/sdr.py data/results/native data/excerpts --baseline data/results/excerpts-deterministic
```

## License

Apache-2.0 ([LICENSE](LICENSE)). Third-party components in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
