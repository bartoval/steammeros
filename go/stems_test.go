package stems

import (
	"errors"
	"math"
	"os"
	"slices"
	"testing"
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
			engine, err := NewEngine(Options{ModelPath: "testdata/" + name + ".onnx", Threads: 1})
			if err != nil {
				t.Fatal(err)
			}
			defer engine.Close()

			stems, err := engine.Separate([][]float32{{0}}, 44100)
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

	stems, err := engine.Separate(audio, 44100)
	if err != nil {
		t.Fatal(err)
	}
	if len(stems) != 4 || stems[3].Name != "vocals" {
		t.Fatalf("unexpected stems: %d", len(stems))
	}
	for _, stem := range stems {
		if len(stem.Channels) != 2 || len(stem.Channels[0]) != frames {
			t.Fatalf("%s: wrong shape", stem.Name)
		}
	}

	if _, err := engine.Separate(audio, 48000); err == nil {
		t.Fatal("48 kHz should be rejected for now")
	}
}
