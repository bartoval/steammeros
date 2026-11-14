// Command separate splits audio files into stems with the native library.
// Development tool: decodes with ffmpeg, writes float32 WAVs.
//
//	go run ./cmd/separate -out stems song.mp3 ...
//
// Output: <out>/<name>/{drums,bass,other,vocals}.wav, where <name> is the file name,
// or the folder name for MUSDB-style <track>/mixture.wav inputs.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/bartoval/steammeros/internal/audiofile"
	"github.com/bartoval/steammeros/stems"
)

func main() {
	model := flag.String("model", "models/scnet-large/scnet-large.onnx", "path to the .onnx model")
	out := flag.String("out", "stems-out", "output folder")
	threads := flag.Int("threads", 0, "inference threads (0 = automatic; 1 = sequential pipeline)")
	cuda := flag.Int("cuda", -1, "run the model on this GPU (needs a GPU build of ONNX Runtime); -1 = CPU")
	ort := flag.String("ort", "", "ONNX Runtime shared library (default: $STEMS_ORT_LIBRARY, then system locations)")
	flag.Parse()

	// Ctrl-C cancels the separation in progress instead of killing the process mid-write.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	start := time.Now()
	engine, err := stems.NewEngine(stems.Options{
		ModelPath: *model, Threads: *threads, CUDA: *cuda >= 0, CUDADevice: *cuda, ORTLibrary: *ort,
	})
	if err != nil {
		fail(err)
	}
	defer engine.Close()
	fmt.Printf("model loaded in %.2f s\n", time.Since(start).Seconds())

	for _, input := range flag.Args() {
		audio, sampleRate, err := audiofile.Decode(ctx, input)
		if err != nil {
			fail(err)
		}

		start := time.Now()
		name := outputName(input)
		showProgress := func(fraction float64) { fmt.Fprintf(os.Stderr, "\r%s: %3.0f%%", name, 100*fraction) }
		separated, err := engine.Separate(ctx, audio, sampleRate, showProgress)
		fmt.Fprintln(os.Stderr)
		if err != nil {
			fail(fmt.Errorf("%s: %w", input, err))
		}
		elapsed := time.Since(start).Seconds()

		dir := filepath.Join(*out, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			fail(err)
		}
		for _, stem := range separated {
			if err := audiofile.WriteWav(filepath.Join(dir, stem.Name+".wav"), stem.Channels, sampleRate); err != nil {
				fail(err)
			}
		}

		seconds := float64(len(audio[0])) / float64(sampleRate)
		fmt.Printf("%s: %.1f s of audio in %.1f s (RTF %.3f)\n", name, seconds, elapsed, elapsed/seconds)
	}
}

// MUSDB tracks are all called mixture.wav, so those are named after their folder.
func outputName(path string) string {
	if filepath.Base(path) == "mixture.wav" {
		return filepath.Base(filepath.Dir(path))
	}
	return strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
