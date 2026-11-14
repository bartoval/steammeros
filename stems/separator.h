#pragma once

#include <cstdint>
#include <vector>

#include "job.h"
#include "scnet.h"

namespace stems {

// Whether a track at `samplerate` can go to the model's 44.1 kHz and back: 8 to 384 kHz,
// with a ratio to 44100 that keeps the resampler small (so 48000 yes, 44056 no).
bool samplerate_supported(int samplerate);

// Separates a whole track the way the original inference script does
// (Seperator.separate_music_file + apply_model with shifts=0):
// normalize, cut into 20 s chunks with 50% overlap, run the model on each,
// blend the chunks with triangular weights, undo the normalization.
//
// Other sample rates go to 44.1 kHz and back, like the original (see resample.h);
// the round trip can come back a sample shorter, so `frames` may differ from the input.
//
// `audio`: `channels` (1 or 2) planar channels of `frames` each, at `samplerate`.
// `samples`: kStemCount * channels * frames floats, [stem][channel][frame].
struct Separation {
  int64_t frames = 0;
  std::vector<float> samples;
};
// `job` reports progress and can cancel the call (it then throws Cancelled).
Separation separate_track(const ScnetModel& model, const float* audio, int channels, int64_t frames,
                          int samplerate, Job& job);

}  // namespace stems
