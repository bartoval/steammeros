// Package stems separates music into stems (drums, bass, other, vocals) in-process.
//
// cgo compiles the C++ core in this directory. ONNX Runtime is loaded at run time from
// its shared library. Prepare installs the model and runtime; build with -tags
// stems_static to link the runtime statically instead (see README.md).
package stems

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo CXXFLAGS: -std=c++20 -I${SRCDIR}/external/onnxruntime -I${SRCDIR}/external/pocketfft
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
const APIVersion = 5

// Options configures an Engine. Prepare fills the paths for managed installations.
type Options struct {
	ModelPath string // path to the .onnx model
	Threads   int    // inference threads; 0 = automatic, 1 = sequential pipeline

	// CUDA runs the model on an NVIDIA GPU. Prepare with CUDA true installs the
	// required ONNX Runtime, CUDA 13 and cuDNN 9 libraries; the driver is external.
	CUDA       bool
	CUDADevice int // which GPU, when CUDA is set

	// ORTLibrary is the path to ONNX Runtime's shared library (libonnxruntime.so or
	// .dylib). Empty: $STEMS_ORT_LIBRARY, then the usual system locations (Homebrew
	// included). One library per process: naming a different one later fails. Ignored
	// with -tags stems_static.
	ORTLibrary string

	// CUDAOpsLibrary is the path to libstems_cuda_ops.so, needed by the fast GPU model
	// (scnet-large-cuda.onnx). Prepare with Fast true installs it. Requires CUDA (Linux).
	CUDAOpsLibrary string
}

// Engine holds the loaded model. Safe for concurrent use; call Close when done.
type Engine struct {
	mu     sync.Mutex // guards the three fields below
	ptr    *C.stems_engine
	closed bool
	active int // Separate calls using ptr right now
}

// Stem is one separated source, planar: Channels[c][i] is sample i of channel c.
type Stem struct {
	Name     string
	Channels [][]float32
}

// Error is returned when the native library reports a failure. Match its kind with
// errors.Is(err, ErrInvalidArgument) and the other Err values below.
type Error struct {
	Status  int
	Message string
}

func (e *Error) Error() string { return fmt.Sprintf("stems: %s (status %d)", e.Message, e.Status) }

// Is makes errors.Is(err, ErrModel) and friends work on *Error.
func (e *Error) Is(target error) bool { return statusErrors[e.Status] == target }

var (
	ErrInvalidArgument = errors.New("stems: invalid argument") // bad audio, sample rate, options...
	ErrModel           = errors.New("stems: model or runtime failure")
	ErrOutOfMemory     = errors.New("stems: out of memory")
	ErrInternal        = errors.New("stems: internal error")
	ErrClosed          = errors.New("stems: engine closed") // using an Engine after Close
)

var statusErrors = map[int]error{
	int(C.STEMS_ERR_INVALID_ARGUMENT): ErrInvalidArgument,
	int(C.STEMS_ERR_MODEL):            ErrModel,
	int(C.STEMS_ERR_OUT_OF_MEMORY):    ErrOutOfMemory,
	int(C.STEMS_ERR_INTERNAL):         ErrInternal,
}

// Version returns the native library version string.
func Version() string { return C.GoString(C.stems_version_string()) }

// NewEngine loads the model. Fails if the native library speaks a different API version.
func NewEngine(opts Options) (*Engine, error) {
	if v := uint32(C.stems_api_version()); v != APIVersion {
		return nil, fmt.Errorf("stems: native API version %d, Go package expects %d", v, APIVersion)
	}

	path := C.CString(opts.ModelPath)
	defer C.free(unsafe.Pointer(path))
	var ortLibrary *C.char
	if opts.ORTLibrary != "" {
		ortLibrary = C.CString(opts.ORTLibrary)
		defer C.free(unsafe.Pointer(ortLibrary))
	}
	var cudaOpsLibrary *C.char
	if opts.CUDAOpsLibrary != "" {
		cudaOpsLibrary = C.CString(opts.CUDAOpsLibrary)
		defer C.free(unsafe.Pointer(cudaOpsLibrary))
	}

	options := C.stems_engine_options{
		struct_size:      C.uint32_t(unsafe.Sizeof(C.stems_engine_options{})),
		model_path:       path,
		threads:          C.int32_t(opts.Threads),
		use_cuda:         C.int32_t(boolToInt(opts.CUDA)),
		cuda_device:      C.int32_t(opts.CUDADevice),
		ort_library:      ortLibrary,
		cuda_ops_library: cudaOpsLibrary,
	}
	var engine *C.stems_engine
	var cerr C.stems_error
	if C.stems_engine_create(&options, &engine, &cerr) != C.STEMS_OK {
		return nil, toError(&cerr)
	}
	return &Engine{ptr: engine}, nil
}

// Close frees the model. It doesn't wait: Separate calls already running finish first,
// then the last one frees it; new calls get ErrClosed. Safe to call twice, and from
// a progress callback.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.closed = true
	e.destroyIfUnused()
	return nil
}

// Marks a Separate call as using the native engine, unless it's closed.
func (e *Engine) acquire() (*C.stems_engine, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, ErrClosed
	}
	e.active++
	return e.ptr, nil
}

func (e *Engine) release() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.active--
	e.destroyIfUnused()
}

// Needs e.mu held.
func (e *Engine) destroyIfUnused() {
	if e.closed && e.active == 0 && e.ptr != nil {
		C.stems_engine_destroy(e.ptr)
		e.ptr = nil
	}
}

// StemNames returns the stems the model produces, in output order. They're fixed, so
// this works on a closed Engine too.
func (e *Engine) StemNames() []string {
	names := make([]string, int(C.stems_engine_stem_count(nil)))
	for i := range names {
		names[i] = C.GoString(C.stems_engine_stem_name(nil, C.int32_t(i)))
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
	engine, err := e.acquire()
	if err != nil {
		return nil, err
	}
	defer e.release()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	planar, frames, err := toPlanar(audio)
	if err != nil {
		return nil, err
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
	status := C.stems_separate(engine, input, C.int32_t(len(audio)), C.int64_t(frames), C.int32_t(sampleRate), job, &result, &cerr)
	close(stop)
	<-stopped
	if status == C.STEMS_ERR_CANCELLED && ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if status != C.STEMS_OK {
		return nil, toError(&cerr)
	}
	defer C.stems_result_destroy(result)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if progress != nil {
		progress(1)
	}
	return copyResult(result), nil
}

// The C side wants one contiguous planar buffer. We copy into it instead of handing over
// the channel slices: a Go slice of Go slices can't be passed to C.
func toPlanar(audio [][]float32) ([]float32, int, error) {
	if len(audio) == 0 {
		return nil, 0, &Error{Status: int(C.STEMS_ERR_INVALID_ARGUMENT), Message: "no channels"}
	}
	frames := len(audio[0])
	planar := make([]float32, 0, len(audio)*frames)
	for c, channel := range audio {
		if len(channel) != frames {
			message := fmt.Sprintf("channel %d has %d samples, channel 0 has %d", c, len(channel), frames)
			return nil, 0, &Error{Status: int(C.STEMS_ERR_INVALID_ARGUMENT), Message: message}
		}
		planar = append(planar, channel...)
	}
	return planar, frames, nil
}

// Copies the native result into Go memory, so it can be freed right away.
func copyResult(result *C.stems_result) []Stem {
	frames := int(C.stems_result_frames(result))
	channels := int(C.stems_result_channels(result))
	stems := make([]Stem, int(C.stems_result_stem_count(result)))
	for s := range stems {
		stems[s].Name = C.GoString(C.stems_engine_stem_name(nil, C.int32_t(s)))
		stems[s].Channels = make([][]float32, channels)
		for c := range channels {
			stems[s].Channels[c] = make([]float32, frames)
			if frames > 0 {
				samples := C.stems_result_samples(result, C.int32_t(s), C.int32_t(c))
				copy(stems[s].Channels[c], unsafe.Slice((*float32)(unsafe.Pointer(samples)), frames))
			}
		}
	}
	return stems
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
