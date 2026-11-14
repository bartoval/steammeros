//go:build stems_static

package stems

// -tags stems_static: links ONNX Runtime statically, for one self-contained binary that
// needs only system libraries. Build it first with CMake (tools/build_ort.sh, then
// cmake), and point pkg-config at it: PKG_CONFIG_PATH=$PWD/build/pkgconfig.

/*
#cgo pkg-config: onnxruntime-static
*/
import "C"
