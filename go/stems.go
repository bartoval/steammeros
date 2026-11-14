// Package stems separates music into stems (drums, bass, other, vocals) in-process.
//
// The native library is linked statically through cgo. Build it first and expose
// it through pkg-config (see README.md):
//
//	tools/build_ort.sh && cmake -S . -B build && cmake --build build
//	export PKG_CONFIG_PATH=$PWD/build/pkgconfig
package stems

/*
#cgo pkg-config: stems
#include <stdlib.h>
#include "stems.h"
*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
	"unsafe"
)

// APIVersion is the C API version this package was written against.
const APIVersion = 3

// Options configures an Engine.
type Options struct {
	ModelPath string // path to the .onnx model
	Threads   int    // inference threads; 0 = automatic, 1 = sequential pipeline

	// CUDA runs the model on an NVIDIA GPU (needs the library built with STEMS_CUDA=ON).
	CUDA       bool
	CUDADevice int // which GPU, when CUDA is set
}

// Engine holds the loaded model. Safe for concurrent use; call Close when done.
type Engine struct {
	mu  sync.RWMutex // Separate and StemNames hold read locks; Close holds the write lock
	ptr *C.stems_engine
}

// Stem is one separated source, planar: Channels[c][i] is sample i of channel c.
type Stem struct {
	Name     string
	Channels [][]float32
}

// Error is returned when the native library reports a failure.
type Error struct {
	Status  int
	Message string
}

func (e *Error) Error() string { return fmt.Sprintf("stems: %s (status %d)", e.Message, e.Status) }

// ErrClosed is returned when using an Engine after Close.
var ErrClosed = errors.New("stems: engine closed")

// Version returns the native library version string.
func Version() string { return C.GoString(C.stems_version_string()) }

// NewEngine loads the model. Fails if the native library speaks a different API version.
func NewEngine(opts Options) (*Engine, error) {
	if v := uint32(C.stems_api_version()); v != APIVersion {
		return nil, fmt.Errorf("stems: native API version %d, Go package expects %d", v, APIVersion)
	}

	path := C.CString(opts.ModelPath)
	defer C.free(unsafe.Pointer(path))

	options := C.stems_engine_options{
		struct_size: C.uint32_t(unsafe.Sizeof(C.stems_engine_options{})),
		model_path:  path,
		threads:     C.int32_t(opts.Threads),
		use_cuda:    C.int32_t(boolToInt(opts.CUDA)),
		cuda_device: C.int32_t(opts.CUDADevice),
	}
	var engine *C.stems_engine
	var cerr C.stems_error
	if C.stems_engine_create(&options, &engine, &cerr) != C.STEMS_OK {
		return nil, toError(&cerr)
	}
	return &Engine{ptr: engine}, nil
}

// Close frees the model. Waits for running Separate calls. Calling it twice is fine.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	C.stems_engine_destroy(e.ptr)
	e.ptr = nil
	return nil
}

// StemNames returns the stems the model produces, in output order.
func (e *Engine) StemNames() []string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	names := make([]string, int(C.stems_engine_stem_count(e.ptr)))
	for i := range names {
		names[i] = C.GoString(C.stems_engine_stem_name(e.ptr, C.int32_t(i)))
	}
	return names
}

// Separate splits audio into stems. `audio` is planar (audio[channel][sample]), 1 or 2
// channels of equal length. The result has the same channels and sample rate; its length
// matches too at 44100 Hz, while other rates can come back a sample shorter.
//
// Blocks until done. Cancelling ctx stops the work within a moment and returns ctx.Err().
// `progress`, if not nil, gets the fraction done (0 to 1) about every 100 ms, from
// another goroutine, and 1 at the end.
func (e *Engine) Separate(ctx context.Context, audio [][]float32, sampleRate int, progress func(float64)) ([]Stem, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.ptr == nil {
		return nil, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(audio) == 0 {
		return nil, &Error{Status: int(C.STEMS_ERR_INVALID_ARGUMENT), Message: "no channels"}
	}

	// The C side wants one contiguous planar buffer. We copy into it instead of handing
	// over the channel slices: a Go slice of Go slices can't be passed to C.
	frames := len(audio[0])
	planar := make([]float32, 0, len(audio)*frames)
	for c, channel := range audio {
		if len(channel) != frames {
			return nil, &Error{Status: int(C.STEMS_ERR_INVALID_ARGUMENT), Message: fmt.Sprintf("channel %d has %d samples, channel 0 has %d", c, len(channel), frames)}
		}
		planar = append(planar, channel...)
	}
	var input *C.float
	if frames > 0 {
		input = (*C.float)(unsafe.Pointer(&planar[0]))
	}

	job := C.stems_job_create()
	if job == nil {
		return nil, &Error{Status: int(C.STEMS_ERR_OUT_OF_MEMORY), Message: "out of memory"}
	}
	defer C.stems_job_destroy(job)

	// The watcher must be gone before the job is freed: we wait for it right after the call.
	stop, stopped := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(stopped)
		watch(ctx, job, progress, stop)
	}()

	var result *C.stems_result
	var cerr C.stems_error
	status := C.stems_separate(e.ptr, input, C.int32_t(len(audio)), C.int64_t(frames), C.int32_t(sampleRate), job, &result, &cerr)
	close(stop)
	<-stopped
	if status == C.STEMS_ERR_CANCELLED && ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if status != C.STEMS_OK {
		return nil, toError(&cerr)
	}
	defer C.stems_result_destroy(result)
	if progress != nil {
		progress(1)
	}

	// Copy the samples into Go memory so the native result can be freed right away.
	outFrames := int(C.stems_result_frames(result))
	stems := make([]Stem, int(C.stems_result_stem_count(result)))
	for s := range stems {
		stems[s].Name = C.GoString(C.stems_engine_stem_name(e.ptr, C.int32_t(s)))
		stems[s].Channels = make([][]float32, len(audio))
		for c := range audio {
			stems[s].Channels[c] = make([]float32, outFrames)
			if outFrames > 0 {
				samples := C.stems_result_samples(result, C.int32_t(s), C.int32_t(c))
				copy(stems[s].Channels[c], unsafe.Slice((*float32)(unsafe.Pointer(samples)), outFrames))
			}
		}
	}
	return stems, nil
}

// Runs next to a stems_separate call until `stop`: cancels the job when ctx is done,
// and reports progress on a ticker.
func watch(ctx context.Context, job *C.stems_job, progress func(float64), stop <-chan struct{}) {
	var tick <-chan time.Time
	if progress != nil {
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		tick = ticker.C
	}
	for {
		select {
		case <-stop:
			return
		case <-ctx.Done():
			C.stems_job_cancel(job)
			return
		case <-tick:
			progress(float64(C.stems_job_progress(job)))
		}
	}
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func toError(cerr *C.stems_error) error {
	return &Error{Status: int(cerr.status), Message: C.GoString(&cerr.message[0])}
}
