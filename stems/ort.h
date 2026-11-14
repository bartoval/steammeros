#pragma once

#include <string>

namespace stems {

// Makes ONNX Runtime's API usable by the rest of the code, once per process.
//
// Static build (CMake, STEMS_ORT_DYNAMIC unset): ONNX Runtime is linked in, `library`
// is ignored. Dynamic build (the Go module): loads the shared library at `library`;
// when empty, $STEMS_ORT_LIBRARY, then the usual system names. Later calls reuse it, and
// fail if they name a different library. Throws Ort::Exception when it can't load.
void load_onnxruntime(const std::string& library);

}  // namespace stems
