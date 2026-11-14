// Package audiofile reads any audio file through ffmpeg and writes float WAVs, for the
// command and the example in this repository. The stems package itself needs neither.
package audiofile

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// Decode reads any audio file ffmpeg understands into planar float32 (audio[channel][sample]),
// keeping its sample rate and channels.
func Decode(path string) ([][]float32, int, error) {
	probe, err := exec.Command("ffprobe", "-v", "error", "-select_streams", "a:0",
		"-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", path).Output()
	if err != nil {
		return nil, 0, fmt.Errorf("ffprobe %s: %w", path, err)
	}
	fields := strings.Split(strings.TrimSpace(string(probe)), ",")
	if len(fields) != 2 {
		return nil, 0, fmt.Errorf("ffprobe %s: no audio stream", path)
	}
	sampleRate, _ := strconv.Atoi(fields[0])
	channels, _ := strconv.Atoi(fields[1])
	if sampleRate <= 0 || channels <= 0 {
		return nil, 0, fmt.Errorf("ffprobe %s: unexpected output %q", path, probe)
	}

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

// WriteWav writes a 32-bit float WAV (format tag 3), interleaving the planar channels.
// Streams through a small buffer instead of building the whole file in memory.
func WriteWav(path string, channels [][]float32, sampleRate int) error {
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
