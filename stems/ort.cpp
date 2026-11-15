#include "ort.h"

#include <cstdlib>
#include <filesystem>
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

// Prepare installs the CUDA runtime beside ORT. Load these libraries by absolute
// path in dependency order: CUDA providers don't all have an $ORIGIN runpath.
// Handles stay loaded just like ORT; no process-wide environment changes needed.
void load_bundled_cuda(const std::string& library) {
#ifdef __linux__
  const auto directory = std::filesystem::path(library).parent_path();
  if (directory.empty()) return;
  for (const char* name : {"libcudart.so.13", "libnvrtc.so.13", "libcublasLt.so.13", "libcublas.so.13",
                           "libcurand.so.10", "libcudnn.so.9"}) {
    const auto path = directory / name;
    if (std::filesystem::exists(path) && !dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL)) {
      throw Ort::Exception("can't load bundled CUDA library " + path.string() + ": " + dlerror(), ORT_FAIL);
    }
  }
#endif
}

std::vector<std::string> candidates(const std::string& library) {
  if (!library.empty()) return {library};
  if (const char* env = std::getenv("STEMS_ORT_LIBRARY"); env && *env) return {env};
  return {std::begin(kDefaultLibraries), std::end(kDefaultLibraries)};
}

// The library stays loaded until the process exits: ORT objects may live that long.
const OrtApiBase* open_api_base(const std::string& library) {
  std::string last_error = "no candidates";
  for (const std::string& name : candidates(library)) {
    load_bundled_cuda(name);
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

  // Official ORT binaries include an uploader. Suppress its initialization before
  // loading ORT, including cached telemetry from earlier runs of other programs.
  if (setenv("ORT_DISABLE_TELEMETRY", "1", 1) != 0) {
    throw Ort::Exception("can't disable ONNX Runtime telemetry", ORT_FAIL);
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
