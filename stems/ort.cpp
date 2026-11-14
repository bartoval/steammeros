#include "ort.h"

#include <cstdlib>
#include <iterator>
#include <mutex>
#include <vector>

#include "onnxruntime_cxx_api.h"

#ifdef STEMS_ORT_DYNAMIC
#include <dlfcn.h>
#endif

namespace stems {
namespace {

std::mutex load_mutex;
bool loaded = false;
std::string loaded_request;  // the path the first load asked for ("" = default search)

#ifdef STEMS_ORT_DYNAMIC

// Tried in order when no path is given. Bare names go through dlopen's own search
// (LD_LIBRARY_PATH, ldconfig, ...); macOS doesn't search Homebrew's prefixes by itself.
const char* const kDefaultLibraries[] = {
    "libonnxruntime.so.1",
    "libonnxruntime.so",
    "libonnxruntime.dylib",
    "/opt/homebrew/lib/libonnxruntime.dylib",
    "/usr/local/lib/libonnxruntime.dylib",
};

std::vector<std::string> candidates(const std::string& library) {
  if (!library.empty()) return {library};
  if (const char* env = std::getenv("STEMS_ORT_LIBRARY"); env && *env) return {env};
  return {std::begin(kDefaultLibraries), std::end(kDefaultLibraries)};
}

// The library stays loaded until the process exits: ORT objects may live that long.
const OrtApiBase* open_api_base(const std::string& library) {
  std::string last_error = "no candidates";
  for (const std::string& name : candidates(library)) {
    void* handle = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      last_error = dlerror();
      continue;
    }
    using GetApiBase = const OrtApiBase* (*)();
    if (auto get_api_base = reinterpret_cast<GetApiBase>(dlsym(handle, "OrtGetApiBase"))) {
      return get_api_base();
    }
    last_error = name + " has no OrtGetApiBase";
  }
  throw Ort::Exception(
      "ONNX Runtime not found (set its path in the options or STEMS_ORT_LIBRARY): " + last_error, ORT_FAIL);
}

#else

const OrtApiBase* open_api_base(const std::string&) { return OrtGetApiBase(); }

#endif

}  // namespace

void load_onnxruntime(const std::string& library) {
  std::lock_guard lock(load_mutex);
  if (loaded) {
#ifdef STEMS_ORT_DYNAMIC
    // One ONNX Runtime per process: asking for another one is a mistake, not a no-op.
    if (!library.empty() && library != loaded_request) {
      const std::string current = loaded_request.empty() ? "the default search" : loaded_request;
      throw Ort::Exception("ONNX Runtime is already loaded from " + current + ", can't also load " + library,
                           ORT_FAIL);
    }
#endif
    return;
  }

  const OrtApiBase* base = open_api_base(library);
  // Returns null when the runtime is older than the headers we compiled against.
  const OrtApi* api = base->GetApi(ORT_API_VERSION);
  if (!api) {
    throw Ort::Exception(
        std::string("ONNX Runtime ") + base->GetVersionString() + " is too old: 1.31 or newer is needed",
        ORT_FAIL);
  }
#ifdef STEMS_ORT_DYNAMIC
  Ort::InitApi(api);
#endif
  loaded = true;
  loaded_request = library;
}

}  // namespace stems
