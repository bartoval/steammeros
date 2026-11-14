/* stems: in-process music source separation, public C API.
 *
 * Audio is float32 PCM, planar: all samples of channel 0, then all of channel 1.
 * No function throws; every failure comes back as a stems_status plus a message.
 */
#ifndef STEMS_H
#define STEMS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version of the C ABI declared by this header. Bumped on breaking changes. */
#define STEMS_API_VERSION 2u

typedef enum stems_status {
  STEMS_OK = 0,
  STEMS_ERR_INVALID_ARGUMENT = 1, /* bad pointer, size, sample rate, channel count, NaN/Inf... */
  STEMS_ERR_MODEL = 2,            /* model file missing/corrupt, or the runtime failed */
  STEMS_ERR_OUT_OF_MEMORY = 3,
  STEMS_ERR_INTERNAL = 4,
} stems_status;

/* Filled in by functions that can fail (optional: pass NULL if you don't need it).
 * Plain struct on purpose: no thread-local "last error", which Go can't use safely
 * because goroutines hop between OS threads. */
typedef struct stems_error {
  stems_status status;
  char message[256];
} stems_error;

typedef struct stems_engine stems_engine;
typedef struct stems_result stems_result;

typedef struct stems_engine_options {
  uint32_t struct_size;     /* sizeof(stems_engine_options), lets us add fields later */
  const char* model_path;   /* path to the .onnx model */
  int32_t threads;          /* threads used by one separation; 0 = one per physical core */
} stems_engine_options;

/* Returns STEMS_API_VERSION of the linked library. */
uint32_t stems_api_version(void);

/* Static version string, valid for the whole process. */
const char* stems_version_string(void);

/* Loads the model. The engine is read-only afterwards and can be shared between threads. */
stems_status stems_engine_create(const stems_engine_options* options, stems_engine** engine, stems_error* error);

/* Frees the engine. Results created by it stay valid. NULL is a no-op. */
void stems_engine_destroy(stems_engine* engine);

/* Number of stems the model produces and their names ("drums", "bass", "other", "vocals"). */
int32_t stems_engine_stem_count(const stems_engine* engine);
const char* stems_engine_stem_name(const stems_engine* engine, int32_t stem);

/* Separates `frames` samples of 1 or 2 channels. Blocks until done.
 * `audio` is planar (channels * frames floats) and is only read during the call.
 * Safe to call from several threads at once on the same engine. */
stems_status stems_separate(stems_engine* engine, const float* audio, int32_t channels, int64_t frames,
                            int32_t samplerate, stems_result** result, stems_error* error);

/* Output layout: for each stem, for each channel, `frames` samples.
 * Same channel count, sample rate and length as the input. */
int32_t stems_result_stem_count(const stems_result* result);
int32_t stems_result_channels(const stems_result* result);
int64_t stems_result_frames(const stems_result* result);
const float* stems_result_samples(const stems_result* result, int32_t stem, int32_t channel);

/* Frees the result and its samples. NULL is a no-op. */
void stems_result_destroy(stems_result* result);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* STEMS_H */
