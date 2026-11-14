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
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

// APIVersion is the C API version this package was written against.
const APIVersion = 2

// Options configures an Engine.
type Options struct {
	ModelPath string // path to the .onnx model
	Threads   int    // threads for one separation; 0 = one per physical core
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
// channels of equal length. The result has the same channels, length and sample rate.
// Blocks until done.
func (e *Engine) Separate(audio [][]float32, sampleRate int) ([]Stem, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.ptr == nil {
		return nil, ErrClosed
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

	var result *C.stems_result
	var cerr C.stems_error
	status := C.stems_separate(e.ptr, input, C.int32_t(len(audio)), C.int64_t(frames), C.int32_t(sampleRate), &result, &cerr)
	if status != C.STEMS_OK {
		return nil, toError(&cerr)
	}
	defer C.stems_result_destroy(result)

	// Copy the samples into Go memory so the native result can be freed right away.
	stems := make([]Stem, int(C.stems_result_stem_count(result)))
	for s := range stems {
		stems[s].Name = C.GoString(C.stems_engine_stem_name(e.ptr, C.int32_t(s)))
		stems[s].Channels = make([][]float32, len(audio))
		for c := range audio {
			stems[s].Channels[c] = make([]float32, frames)
			if frames > 0 {
				samples := C.stems_result_samples(result, C.int32_t(s), C.int32_t(c))
				copy(stems[s].Channels[c], unsafe.Slice((*float32)(unsafe.Pointer(samples)), frames))
			}
		}
	}
	return stems, nil
}

func toError(cerr *C.stems_error) error {
	return &Error{Status: int(cerr.status), Message: C.GoString(&cerr.message[0])}
}
