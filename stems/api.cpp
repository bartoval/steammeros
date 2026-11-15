// The C boundary: validates arguments, turns C++ exceptions into stems_status codes.
// Nothing may throw past this file.
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "ort.h"
#include "scnet.h"
#include "separator.h"
#include "stems.h"

struct stems_engine {
  std::unique_ptr<stems::ScnetModel> model;
};

struct stems_job {
  stems::Job job;
};

struct stems_result {
  int32_t channels = 0;
  int64_t frames = 0;
  std::vector<float> samples;  // [stem][channel][frame]
};

namespace {

// Thrown inside this file for bad input; mapped to STEMS_ERR_INVALID_ARGUMENT.
struct InvalidArgument : std::runtime_error {
  using std::runtime_error::runtime_error;
};

stems_status fail(stems_error* error, stems_status status, const char* message) {
  if (error) {
    error->status = status;
    std::snprintf(error->message, sizeof(error->message), "%s", message);
  }
  return status;
}

stems_status succeed(stems_error* error) {
  if (error) {
    error->status = STEMS_OK;
    error->message[0] = '\0';
  }
  return STEMS_OK;
}

// Runs `body` and converts whatever it throws into a status + message.
template <typename Body>
stems_status guarded(stems_error* error, Body&& body) {
  try {
    body();
    return succeed(error);
  } catch (const InvalidArgument& e) {
    return fail(error, STEMS_ERR_INVALID_ARGUMENT, e.what());
  } catch (const stems::Cancelled&) {
    return fail(error, STEMS_ERR_CANCELLED, "cancelled");
  } catch (const std::bad_alloc&) {
    return fail(error, STEMS_ERR_OUT_OF_MEMORY, "out of memory");
  } catch (const Ort::Exception& e) {
    return fail(error, STEMS_ERR_MODEL, e.what());
  } catch (const std::exception& e) {
    return fail(error, STEMS_ERR_INTERNAL, e.what());
  } catch (...) {
    return fail(error, STEMS_ERR_INTERNAL, "unknown error");
  }
}

void check_audio(const float* audio, int32_t channels, int64_t frames, int32_t samplerate) {
  if (channels != 1 && channels != 2) throw InvalidArgument("channels must be 1 or 2");
  if (frames < 0) throw InvalidArgument("frames must be >= 0");
  if (frames > std::numeric_limits<int64_t>::max() / (stems::kStemCount * 2)) {
    throw InvalidArgument("too many frames");
  }
  if (frames > 0 && !audio) throw InvalidArgument("audio is NULL");
  if (!stems::samplerate_supported(samplerate)) {
    throw InvalidArgument(
        "unsupported sample rate: use 8000-384000 Hz with a simple ratio to 44100 (e.g. 48000)");
  }

  for (int64_t i = 0; i < channels * frames; ++i) {
    if (!std::isfinite(audio[i])) throw InvalidArgument("audio contains NaN or Inf");
  }
}

}  // namespace

extern "C" {

uint32_t stems_api_version(void) { return STEMS_API_VERSION; }

const char* stems_version_string(void) { return "stems " STEMS_VERSION; }

stems_status stems_engine_create(const stems_engine_options* options, stems_engine** engine,
                                 stems_error* error) {
  if (!engine) return fail(error, STEMS_ERR_INVALID_ARGUMENT, "engine is NULL");
  *engine = nullptr;
  if (!options || options->struct_size < sizeof(stems_engine_options)) {
    return fail(error, STEMS_ERR_INVALID_ARGUMENT, "options missing or struct_size too small");
  }
  if (!options->model_path) return fail(error, STEMS_ERR_INVALID_ARGUMENT, "model_path is NULL");
  const char* cuda_ops = options->cuda_ops_library ? options->cuda_ops_library : "";
  if (*cuda_ops && (!options->use_cuda || options->cuda_device < 0)) {
    return fail(error, STEMS_ERR_INVALID_ARGUMENT, "cuda_ops_library requires use_cuda");
  }

  return guarded(error, [&] {
    auto created = std::make_unique<stems_engine>();
    const int cuda_device = options->use_cuda ? options->cuda_device : -1;
    stems::load_onnxruntime(options->ort_library ? options->ort_library : "");
    created->model =
        std::make_unique<stems::ScnetModel>(options->model_path, options->threads, cuda_device, cuda_ops);
    *engine = created.release();
  });
}

void stems_engine_destroy(stems_engine* engine) { delete engine; }

int32_t stems_engine_stem_count(const stems_engine*) { return stems::kStemCount; }

const char* stems_engine_stem_name(const stems_engine*, int32_t stem) {
  if (stem < 0 || stem >= stems::kStemCount) return nullptr;
  return stems::kStemNames[stem];
}

stems_job* stems_job_create(void) {
  try {
    return new stems_job();
  } catch (...) {
    return nullptr;
  }
}

void stems_job_cancel(stems_job* job) {
  if (job) job->job.cancel();
}

double stems_job_progress(const stems_job* job) { return job ? job->job.progress() : 0.0; }

void stems_job_destroy(stems_job* job) { delete job; }

stems_status stems_separate(stems_engine* engine, const float* audio, int32_t channels, int64_t frames,
                            int32_t samplerate, stems_job* job, stems_result** result, stems_error* error) {
  if (!result) return fail(error, STEMS_ERR_INVALID_ARGUMENT, "result is NULL");
  *result = nullptr;
  if (!engine) return fail(error, STEMS_ERR_INVALID_ARGUMENT, "engine is NULL");

  return guarded(error, [&] {
    check_audio(audio, channels, frames, samplerate);
    stems_job local;  // no job from the caller: nobody can cancel, progress goes nowhere
    stems::Job& tracker = job ? job->job : local.job;
    tracker.throw_if_cancelled();

    auto separated = std::make_unique<stems_result>();
    stems::Separation separation =
        stems::separate_track(*engine->model, audio, channels, frames, samplerate, tracker);
    separated->channels = channels;
    separated->frames = separation.frames;
    separated->samples = std::move(separation.samples);
    *result = separated.release();
  });
}

int32_t stems_result_stem_count(const stems_result* result) { return result ? stems::kStemCount : 0; }

int32_t stems_result_channels(const stems_result* result) { return result ? result->channels : 0; }

int64_t stems_result_frames(const stems_result* result) { return result ? result->frames : 0; }

const float* stems_result_samples(const stems_result* result, int32_t stem, int32_t channel) {
  if (!result || stem < 0 || stem >= stems::kStemCount || channel < 0 || channel >= result->channels) {
    return nullptr;
  }
  if (result->frames == 0) return nullptr;
  return result->samples.data() + (static_cast<int64_t>(stem) * result->channels + channel) * result->frames;
}

void stems_result_destroy(stems_result* result) { delete result; }

}  // extern "C"
