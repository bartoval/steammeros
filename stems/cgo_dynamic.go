//go:build !stems_static

package stems

// Default build: ONNX Runtime is loaded at run time (ort.cpp), so `go get` is all it takes
// to compile. Needs the shared library at run time: see Options.ORTLibrary.

/*
#cgo CXXFLAGS: -DSTEMS_ORT_DYNAMIC -DORT_API_MANUAL_INIT
#cgo linux LDFLAGS: -ldl -lstdc++ -lm
#cgo darwin LDFLAGS: -lc++
*/
import "C"
