package stems

import (
	"context"
	"errors"
	"math"
	"os"
	"slices"
	"testing"
	"time"
)

const testModel = "../models/scnet-large/scnet-large.onnx"

func TestStemNamesConcurrentClose(t *testing.T) {
	engine, err := NewEngine(Options{ModelPath: "testdata/wrong-output-shape.onnx", Threads: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()

	start := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		<-start
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	}()

	close(start)
	want := []string{"drums", "bass", "other", "vocals"}
	for i := 0; i < 100; i++ {
		if got := engine.StemNames(); !slices.Equal(got, want) {
			t.Errorf("unexpected stem names: %v", got)
			break
		}
	}
	<-done
}

// Tiny ONNX fixtures exercise the native output checks without the SCNet weights.
func TestSeparateRejectsIncompatibleOutput(t *testing.T) {
	for _, name := range []string{"wrong-output-shape", "wrong-output-type"} {
		t.Run(name, func(t *testing.T) {
			engine, err := NewEngine(Options{ModelPath: "testdata/" + name + ".onnx", Threads: 2})
			if err != nil {
				t.Fatal(err)
			}
			defer engine.Close()

			// Fail while the following chunk is being prepared: native workers must
			// finish before Separate returns the model error and releases buffers.
			stems, err := engine.Separate(context.Background(), [][]float32{make([]float32, 10*44100+1)}, 44100, nil)
			var nativeErr *Error
			if !errors.As(err, &nativeErr) || nativeErr.Status != 2 { // STEMS_ERR_MODEL
				t.Fatalf("expected model error, got %v", err)
			}
			if stems != nil {
				t.Fatal("incompatible output returned stems")
			}
		})
	}
}

// The fixture repeats the input spectrum into all four stems. Three chunks of
// different lengths exercise pipeline draining and ordered overlap-add cheaply.
func TestSeparatePipelineMatchesSerial(t *testing.T) {
	serial, err := NewEngine(Options{ModelPath: "testdata/repeat-spectrum.onnx", Threads: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer serial.Close()
	parallel, err := NewEngine(Options{ModelPath: "testdata/repeat-spectrum.onnx", Threads: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer parallel.Close()

	for _, tc := range []struct {
		name     string
		channels int
		frames   int
	}{
		{"empty", 1, 0},
		{"short", 1, 1},
		{"mono-overlap", 1, 10*44100 + 1},
		{"stereo-overlap", 2, 20*44100 + 173},
	} {
		t.Run(tc.name, func(t *testing.T) {
			audio := make([][]float32, tc.channels)
			for c := range audio {
				audio[c] = make([]float32, tc.frames)
				for i := range audio[c] {
					audio[c][i] = float32((i*(c+1))%4093-2046) / 8192
				}
			}
			want, err := serial.Separate(context.Background(), audio, 44100, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := parallel.Separate(context.Background(), audio, 44100, nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("got %d stems, want %d", len(got), len(want))
			}
			for s := range want {
				if got[s].Name != want[s].Name || len(got[s].Channels) != tc.channels {
					t.Fatalf("stem %d: wrong name or channel count", s)
				}
				for c := range audio {
					if !slices.Equal(got[s].Channels[c], want[s].Channels[c]) {
						t.Fatalf("%s channel %d: pipeline differs from serial", got[s].Name, c)
					}
					if len(got[s].Channels[c]) != tc.frames {
						t.Fatalf("%s channel %d: wrong length", got[s].Name, c)
					}
					for i, sample := range got[s].Channels[c] {
						if math.Abs(float64(sample-audio[c][i])) > 2e-6 {
							t.Fatalf("%s channel %d sample %d: round trip differs", got[s].Name, c, i)
						}
					}
				}
			}
		})
	}
}

// Runs the real model on 3 s of noise and checks the shape of the result.
// Skipped when the model file isn't there (it's not in Git).
func TestSeparate(t *testing.T) {
	if _, err := os.Stat(testModel); err != nil {
		t.Skip("model not found:", testModel)
	}
	engine, err := NewEngine(Options{ModelPath: testModel})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()

	frames := 3 * 44100
	audio := [][]float32{make([]float32, frames), make([]float32, frames)}
	for i := 0; i < frames; i++ {
		audio[0][i] = float32(0.1 * math.Sin(float64(i)*0.05))
		audio[1][i] = float32(0.1 * math.Sin(float64(i)*0.031))
	}

	var lastProgress float64
	stems, err := engine.Separate(context.Background(), audio, 44100, func(p float64) { lastProgress = p })
	if err != nil {
		t.Fatal(err)
	}
	if lastProgress != 1 {
		t.Fatalf("last progress %v, want 1", lastProgress)
	}
	if len(stems) != 4 || stems[3].Name != "vocals" {
		t.Fatalf("unexpected stems: %d", len(stems))
	}
	for _, stem := range stems {
		if len(stem.Channels) != 2 || len(stem.Channels[0]) != frames {
			t.Fatalf("%s: wrong shape", stem.Name)
		}
	}

	// 48 kHz goes through the resampler; the round trip may drop one sample.
	stems, err = engine.Separate(context.Background(), audio, 48000, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(stems[0].Channels[0]); got < frames-1 || got > frames {
		t.Fatalf("48 kHz: %d frames, want %d or one less", got, frames)
	}
	if _, err := engine.Separate(context.Background(), audio, 44056, nil); err == nil {
		t.Fatal("44056 Hz has no simple ratio to 44100 and should be rejected")
	}
}

// A cancelled context must stop the model mid-chunk: one 20 s chunk takes several
// seconds on CPU, so returning well before that shows ORT's Run was interrupted.
func TestSeparateCancel(t *testing.T) {
	if _, err := os.Stat(testModel); err != nil {
		t.Skip("model not found:", testModel)
	}
	engine, err := NewEngine(Options{ModelPath: testModel})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()

	audio := [][]float32{make([]float32, 60*44100)}
	for i := range audio[0] {
		audio[0][i] = float32(0.1 * math.Sin(float64(i)*0.05))
	}
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err = engine.Separate(ctx, audio, 44100, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v, want context.DeadlineExceeded", err)
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Fatalf("took %v to stop", elapsed)
	}
}
