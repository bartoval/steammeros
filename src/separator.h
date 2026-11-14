#pragma once

#include <cstdint>
#include <vector>

#include "scnet.h"

namespace stems {

// Separates a whole track the way the original inference script does
// (Seperator.separate_music_file + apply_model with shifts=0):
// normalize, cut into 20 s chunks with 50% overlap, run the model on each,
// blend the chunks with triangular weights, undo the normalization.
//
// `audio`: `channels` (1 or 2) planar channels of `samples` each, at 44.1 kHz.
// Returns kStemCount * channels * samples floats, [stem][channel][sample].
std::vector<float> separate_track(const ScnetModel& model, const float* audio, int channels, int64_t samples);

}  // namespace stems
