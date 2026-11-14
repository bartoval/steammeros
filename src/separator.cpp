#include "separator.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <utility>

#include "resample.h"

namespace stems {
namespace {

constexpr int64_t kSegment = 20 * kSamplerate;  // apply_model(segment=20)
constexpr int64_t kStride = kSegment / 2;       // overlap=0.5

// apply_model's blending weight: rises 1..N/2, falls N/2..1, scaled so the peak is 1.
// Where chunks overlap, each one counts proportionally to how far we are from its edges.
std::vector<float> triangle_weights() {
  std::vector<float> weight(kSegment);
  const float peak = kSegment / 2.0f;
  for (int64_t i = 0; i < kSegment; ++i) {
    weight[i] = (i < kSegment / 2 ? i + 1 : kSegment - i) / peak;
  }
  return weight;
}

struct Normalization {
  double mean = 0.0;
  double std = 1.0;
};

// Mean and standard deviation of the mono downmix, over the whole track
// (unbiased std, like torch.std). Silence has std 0: we keep 1 instead of
// dividing by zero (the original would produce NaN there).
Normalization measure(const std::vector<float>& stereo, int64_t samples) {
  double sum = 0.0;
  for (int64_t i = 0; i < samples; ++i) sum += 0.5 * (stereo[i] + stereo[samples + i]);
  const double mean = sum / samples;

  double squares = 0.0;
  for (int64_t i = 0; i < samples; ++i) {
    const double d = 0.5 * (stereo[i] + stereo[samples + i]) - mean;
    squares += d * d;
  }
  const double std = samples > 1 ? std::sqrt(squares / (samples - 1)) : 0.0;
  return {mean, std > 0.0 ? std : 1.0};
}

// Mono input: averages each stem's two channels into one, turning [stem][2][sample]
// into [stem][sample]. Stem s is written where stems < s were read, so in place is safe.
void downmix_in_place(std::vector<float>& stems, int64_t samples) {
  for (int s = 0; s < kStemCount; ++s) {
    const float* left = stems.data() + (s * kModelChannels) * samples;
    const float* right = left + samples;
    float* mono = stems.data() + s * samples;
    for (int64_t i = 0; i < samples; ++i) mono[i] = 0.5f * (left[i] + right[i]);
  }
  stems.resize(kStemCount * samples);
  stems.shrink_to_fit();
}

// The original separation at the model rate: chunk, run, blend, denormalize.
// Returns kStemCount * channels * samples floats, [stem][channel][sample].
std::vector<float> separate_at_model_rate(const ScnetModel& model, const float* audio, int channels,
                                          int64_t samples) {
  if (samples == 0) return {};

  // The model is stereo: mono input is duplicated on both channels.
  std::vector<float> stereo(kModelChannels * samples);
  for (int c = 0; c < kModelChannels; ++c) {
    const float* source = audio + (channels == 1 ? 0 : c) * samples;
    std::copy(source, source + samples, stereo.begin() + c * samples);
  }

  const Normalization norm = measure(stereo, samples);
  for (float& x : stereo) x = static_cast<float>((x - norm.mean) / norm.std);

  // Weighted overlap-add of the chunks.
  const std::vector<float> weight = triangle_weights();
  std::vector<float> sum(kStemCount * kModelChannels * samples, 0.0f);
  std::vector<float> weight_sum(samples, 0.0f);
  const auto prepare = [&](int64_t offset) {
    const int64_t length = std::min(kSegment, samples - offset);

    std::vector<float> chunk_in(kModelChannels * length);
    for (int c = 0; c < kModelChannels; ++c) {
      const float* source = stereo.data() + c * samples + offset;
      std::copy(source, source + length, chunk_in.begin() + c * length);
    }

    return model.prepare(chunk_in.data(), length);
  };

  const auto reconstruct_and_blend = [&](ScnetModel::Output prediction, int64_t offset) {
    const int64_t length = prediction.samples;
    std::vector<float> chunk_out(kStemCount * kModelChannels * length);
    model.reconstruct(prediction, chunk_out.data());
    for (int64_t signal = 0; signal < kStemCount * kModelChannels; ++signal) {
      float* target = sum.data() + signal * samples + offset;
      const float* source = chunk_out.data() + signal * length;
      for (int64_t i = 0; i < length; ++i) target[i] += weight[i] * source[i];
    }
    for (int64_t i = 0; i < length; ++i) weight_sum[offset + i] += weight[i];
  };

  if (!model.pipeline_enabled() || samples <= kStride) {
    for (int64_t offset = 0; offset < samples; offset += kStride) {
      auto input = prepare(offset);
      reconstruct_and_blend(model.infer(input), offset);
    }
  } else {
    // One prepared chunk ahead, one reconstruction behind, one Run at a time.
    // Waiting for the previous reconstruction preserves overlap-add order and
    // bounds retained ORT outputs to two, independent of track length.
    // Futures join on all exits, including exceptions, before captured buffers die.
    auto prepared = std::async(std::launch::async, prepare, int64_t{0});
    std::future<void> reconstructed;
    for (int64_t offset = 0; offset < samples; offset += kStride) {
      auto input = prepared.get();
      if (offset + kStride < samples) {
        prepared = std::async(std::launch::async, prepare, offset + kStride);
      }
      auto prediction = model.infer(input);
      if (reconstructed.valid()) reconstructed.get();
      reconstructed = std::async(std::launch::async, reconstruct_and_blend, std::move(prediction), offset);
    }
    reconstructed.get();
  }

  stereo = {};  // the normalized input isn't needed anymore

  // Divide by the total weight and undo the normalization. Done in place: `sum`
  // already has the [stem][channel][sample] layout, so it becomes the result.
  for (int64_t signal = 0; signal < kStemCount * kModelChannels; ++signal) {
    float* stem = sum.data() + signal * samples;
    for (int64_t i = 0; i < samples; ++i) {
      stem[i] = static_cast<float>(stem[i] / weight_sum[i] * norm.std + norm.mean);
    }
  }
  if (channels == 1) downmix_in_place(sum, samples);
  return sum;
}

// Resamples each of `signals` planar channels, one thread per channel.
std::vector<float> resample_planar(const Resampler& resampler, const float* audio, int signals,
                                   int64_t frames) {
  const int64_t out_frames = resampler.output_length(frames);
  std::vector<float> out(signals * out_frames);
  std::vector<std::future<void>> jobs;
  jobs.reserve(signals);
  for (int s = 0; s < signals; ++s) {
    jobs.push_back(std::async(std::launch::async, [&, s] {
      const std::vector<float> resampled = resampler.run(audio + s * frames, frames);
      std::copy(resampled.begin(), resampled.end(), out.begin() + s * out_frames);
    }));
  }
  for (auto& job : jobs) job.get();
  return out;
}

}  // namespace

// Same path as the original inference: to 44.1 kHz, separate, back to the input rate.
Separation separate_track(const ScnetModel& model, const float* audio, int channels, int64_t frames,
                          int samplerate) {
  if (samplerate == kSamplerate) return {frames, separate_at_model_rate(model, audio, channels, frames)};

  const Resampler to_model(samplerate, kSamplerate);
  const Resampler back(kSamplerate, samplerate);
  const int64_t model_frames = to_model.output_length(frames);

  std::vector<float> input = resample_planar(to_model, audio, channels, frames);
  const std::vector<float> separated = separate_at_model_rate(model, input.data(), channels, model_frames);
  input = {};
  return {back.output_length(model_frames),
          resample_planar(back, separated.data(), kStemCount * channels, model_frames)};
}

}  // namespace stems
