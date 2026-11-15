//go:build stems_cuda_test

package stems

import (
	"context"
	"errors"
	"math"
	"os"
	"testing"
	"time"
)

// Opt-in integration check with an actual CUDA-exported model and operator library.
// Small, different input lengths exercise workspace reuse across concurrent Runs.
func TestCUDAOperatorsConcurrentAndCancel(t *testing.T) {
	model, operators := os.Getenv("STEMS_TEST_CUDA_MODEL"), os.Getenv("STEMS_TEST_CUDA_OPS")
	if model == "" || operators == "" {
		t.Skip("set STEMS_TEST_CUDA_MODEL and STEMS_TEST_CUDA_OPS")
	}
	engine, err := NewEngine(Options{ModelPath: model, CUDA: true, CUDAOpsLibrary: operators, Threads: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	inputs := make([][][]float32, 2)
	want := make([][]Stem, len(inputs))
	for i, frames := range []int{8192, 13312} {
		inputs[i] = [][]float32{make([]float32, frames), make([]float32, frames)}
		for c := range inputs[i] {
			for n := range inputs[i][c] {
				inputs[i][c][n] = float32(0.1 * math.Sin(float64(n)*0.031*float64(c+1)))
			}
		}
		want[i], err = engine.Separate(context.Background(), inputs[i], 44100, nil)
		if err != nil {
			t.Fatal(err)
		}
	}
	type result struct {
		index int
		stems []Stem
		err   error
	}
	checkResult := func(got result) {
		t.Helper()
		if got.err != nil {
			t.Fatal(got.err)
		}
		if len(got.stems) != len(want[got.index]) {
			t.Fatal("wrong number of stems")
		}
		for s := range got.stems {
			if len(got.stems[s].Channels) != len(want[got.index][s].Channels) {
				t.Fatal("wrong number of channels")
			}
			for c := range want[got.index][s].Channels {
				if len(got.stems[s].Channels[c]) != len(want[got.index][s].Channels[c]) {
					t.Fatal("wrong number of frames")
				}
				var energy, errorEnergy, maxError float64
				for n, value := range got.stems[s].Channels[c] {
					ref := float64(want[got.index][s].Channels[c][n])
					delta := float64(value) - ref
					if math.IsNaN(delta) || math.IsInf(delta, 0) {
						t.Fatal("non-finite output")
					}
					energy += ref * ref
					errorEnergy += delta * delta
					maxError = math.Max(maxError, math.Abs(delta))
				}
				// CUDA's ordinary FP32 graph also varies between Runs. Allow
				// tiny rounding differences, but require 80 dB agreement.
				if errorEnergy > energy*1e-8 || maxError > 1e-6 {
					t.Fatalf("repeated Run changed input %d, stem %d, channel %d: SNR %.2f dB, max %.9g", got.index, s, c, 10*math.Log10(energy/errorEnergy), maxError)
				}
			}
		}
	}

	done := make(chan result, len(inputs))
	start := make(chan struct{})
	for i := range inputs {
		go func(i int) {
			<-start
			got, err := engine.Separate(context.Background(), inputs[i], 44100, nil)
			done <- result{i, got, err}
		}(i)
	}
	close(start)
	for range inputs {
		select {
		case got := <-done:
			checkResult(got)
		case <-time.After(30 * time.Second):
			t.Fatal("concurrent Run did not finish")
		}
	}

	// Cancellation must leave the engine usable for the following call.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	timer := time.AfterFunc(100*time.Millisecond, cancel)
	defer timer.Stop()
	cancelStarted := time.Now()
	_, err = engine.Separate(ctx, [][]float32{make([]float32, 40*44100)}, 44100, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if elapsed := time.Since(cancelStarted); elapsed > 3*time.Second {
		t.Fatalf("took %v to stop", elapsed)
	}
	t.Logf("cancelled call returned in %v", time.Since(cancelStarted))
	after, err := engine.Separate(context.Background(), inputs[0], 44100, nil)
	checkResult(result{0, after, err})
}
