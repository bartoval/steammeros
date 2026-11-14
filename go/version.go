// Package stems is the in-process Go binding of the stems native library.
//
// The native code is linked statically through cgo. Build the native archive
// first and expose it through pkg-config (see README.md):
//
//	cmake -S . -B build && cmake --build build
//	export PKG_CONFIG_PATH=$PWD/build/pkgconfig
package stems

/*
#cgo pkg-config: stems
#include "stems.h"
*/
import "C"

// APIVersion is the C API version this package was written against.
const APIVersion = 1

// NativeAPIVersion returns the C API version reported by the linked library.
func NativeAPIVersion() uint32 { return uint32(C.stems_api_version()) }

// Version returns the native library version string.
func Version() string { return C.GoString(C.stems_version_string()) }
