#pragma once

#include <string>

namespace stems {

// Makes ONNX Runtime's API usable by the rest of the code, once per process.
//
// Static build (CMake, STEMS_ORT_DYNAMIC unset): ONNX Runtime is linked in, `library`
// is ignored. Dynamic build (the Go module): loads the shared library at `library`;
// when empty, $STEMS_ORT_LIBRARY, then the usual system names. The first successful
// load wins for the whole process. Throws Ort::Exception when it can't load.
void load_onnxruntime(const std::string& library);

// Whether load_onnxruntime() has succeeded, so the Ort:: classes can be used.
bool onnxruntime_loaded();

}  // namespace stems
