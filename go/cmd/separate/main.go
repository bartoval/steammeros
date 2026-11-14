// Command separate splits audio files into stems with the native library.
// Development tool: decodes with ffmpeg, writes float32 WAVs.
//
//	go run ./cmd/separate -model ../models/scnet-large/scnet-large.onnx -out ../data/results/native song.wav ...
//
// Output: <out>/<name>/{drums,bass,other,vocals}.wav, where <name> is the file name,
// or the folder name for MUSDB-style <track>/mixture.wav inputs.
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"steammeros/stems"
)

func main() {
	model := flag.String("model", "models/scnet-large/scnet-large.onnx", "path to the .onnx model")
	out := flag.String("out", "stems-out", "output folder")
	threads := flag.Int("threads", 0, "inference threads (0 = automatic; 1 = sequential pipeline)")
	cuda := flag.Int("cuda", -1, "run the model on this GPU (needs a STEMS_CUDA build); -1 = CPU")
	flag.Parse()

	// Ctrl-C cancels the separation in progress instead of killing the process mid-write.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	start := time.Now()
	engine, err := stems.NewEngine(stems.Options{ModelPath: *model, Threads: *threads, CUDA: *cuda >= 0, CUDADevice: *cuda})
	if err != nil {
		fail(err)
	}
	defer engine.Close()
	fmt.Printf("model loaded in %.2f s\n", time.Since(start).Seconds())

	for _, input := range flag.Args() {
		audio, sampleRate, err := decode(input)
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
			if err := writeWav(filepath.Join(dir, stem.Name+".wav"), stem.Channels, sampleRate); err != nil {
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

// Decodes any audio file with ffmpeg into planar float32, keeping its sample rate and channels.
func decode(path string) ([][]float32, int, error) {
	probe, err := exec.Command("ffprobe", "-v", "error", "-select_streams", "a:0",
		"-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", path).Output()
	if err != nil {
		return nil, 0, fmt.Errorf("ffprobe %s: %w", path, err)
	}
	fields := strings.Split(strings.TrimSpace(string(probe)), ",")
	if len(fields) != 2 {
		return nil, 0, fmt.Errorf("ffprobe %s: unexpected output %q", path, probe)
	}
	sampleRate, _ := strconv.Atoi(fields[0])
	channels, _ := strconv.Atoi(fields[1])

	raw, err := exec.Command("ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1").Output()
	if err != nil {
		return nil, 0, fmt.Errorf("ffmpeg %s: %w", path, err)
	}

	// ffmpeg gives interleaved samples (L R L R...): split them per channel.
	frames := len(raw) / 4 / channels
	audio := make([][]float32, channels)
	for c := range audio {
		audio[c] = make([]float32, frames)
		for i := 0; i < frames; i++ {
			bits := binary.LittleEndian.Uint32(raw[(i*channels+c)*4:])
			audio[c][i] = math.Float32frombits(bits)
		}
	}
	return audio, sampleRate, nil
}

// Writes a 32-bit float WAV (format tag 3), interleaving the planar channels.
// Streams through a small buffer instead of building the whole file in memory.
func writeWav(path string, channels [][]float32, sampleRate int) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	frames := len(channels[0])
	dataSize := frames * len(channels) * 4
	out := bufio.NewWriter(file)
	put := func(v any) { binary.Write(out, binary.LittleEndian, v) }
	out.WriteString("RIFF")
	put(uint32(36 + dataSize))
	out.WriteString("WAVEfmt ")
	put(uint32(16))                             // fmt chunk size
	put(uint16(3))                              // IEEE float
	put(uint16(len(channels)))                  // channels
	put(uint32(sampleRate))                     // sample rate
	put(uint32(sampleRate * len(channels) * 4)) // bytes per second
	put(uint16(len(channels) * 4))              // bytes per frame
	put(uint16(32))                             // bits per sample
	out.WriteString("data")
	put(uint32(dataSize))

	var sample [4]byte
	for i := 0; i < frames; i++ {
		for c := range channels {
			binary.LittleEndian.PutUint32(sample[:], math.Float32bits(channels[c][i]))
			out.Write(sample[:])
		}
	}
	if err := out.Flush(); err != nil {
		return err
	}
	return file.Close()
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
