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
constexpr int kSignals = kStemCount * kModelChannels;
constexpr int kMinSamplerate = 8000;
constexpr int kMaxSamplerate = 384000;

struct Normalization {
  double mean = 0.0;
  double std = 1.0;
};

// The model's input: always stereo (mono is duplicated), normalized by the mono downmix's
// mean and standard deviation over the whole track (unbiased std, like torch.std).
// Silence has std 0: we keep 1 instead of dividing by zero (the original gives NaN).
struct ModelInput {
  std::vector<float> stereo;  // [channel][sample]
  Normalization norm;
};

ModelInput normalized_stereo(const float* audio, int channels, int64_t samples) {
  ModelInput input{std::vector<float>(kModelChannels * samples), {}};
  for (int c = 0; c < kModelChannels; ++c) {
    const float* source = audio + (channels == 1 ? 0 : c) * samples;
    std::copy(source, source + samples, input.stereo.begin() + c * samples);
  }

  const float* left = input.stereo.data();
  const float* right = left + samples;
  double sum = 0.0;
  for (int64_t i = 0; i < samples; ++i) sum += 0.5 * (left[i] + right[i]);
  const double mean = sum / samples;
  double squares = 0.0;
  for (int64_t i = 0; i < samples; ++i) {
    const double d = 0.5 * (left[i] + right[i]) - mean;
    squares += d * d;
  }
  const double std = samples > 1 ? std::sqrt(squares / (samples - 1)) : 0.0;
  input.norm = {mean, std > 0.0 ? std : 1.0};

  for (float& x : input.stereo) x = static_cast<float>((x - input.norm.mean) / input.norm.std);
  return input;
}

// apply_model's blending: each chunk's stems are added with a triangular weight (rising
// 1..N/2, falling N/2..1, peak 1), then every sample is divided by the weights it got.
// Where chunks overlap, each counts in proportion to how far we are from its edges.
class Blender {
 public:
  explicit Blender(int64_t samples)
      : samples_(samples), weight_(kSegment), sum_(kSignals * samples, 0.0f), weight_sum_(samples, 0.0f) {
    const float peak = kSegment / 2.0f;
    for (int64_t i = 0; i < kSegment; ++i) weight_[i] = (i < kSegment / 2 ? i + 1 : kSegment - i) / peak;
  }

  // `chunk`: kSignals planar signals of `length` samples, starting at `offset`.
  void add(const std::vector<float>& chunk, int64_t length, int64_t offset) {
    for (int signal = 0; signal < kSignals; ++signal) {
      float* target = sum_.data() + signal * samples_ + offset;
      const float* source = chunk.data() + signal * length;
      for (int64_t i = 0; i < length; ++i) target[i] += weight_[i] * source[i];
    }
    for (int64_t i = 0; i < length; ++i) weight_sum_[offset + i] += weight_[i];
  }

  // Divides by the total weight and undoes the normalization, in place: the sum buffer
  // already has the [stem][channel][sample] layout, so it becomes the result.
  std::vector<float> finish(const Normalization& norm) {
    for (int signal = 0; signal < kSignals; ++signal) {
      float* stem = sum_.data() + signal * samples_;
      for (int64_t i = 0; i < samples_; ++i) {
        stem[i] = static_cast<float>(stem[i] / weight_sum_[i] * norm.std + norm.mean);
      }
    }
    return std::move(sum_);
  }

 private:
  int64_t samples_;
  std::vector<float> weight_;
  std::vector<float> sum_;         // [stem][channel][sample]
  std::vector<float> weight_sum_;  // [sample]
};

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
//
// Chunks go through three stages: prepare (STFT), infer (the network), reconstruct and
// blend (iSTFT). With several workers they overlap: one chunk prepared ahead, one
// reconstructed behind, one Run at a time. With one worker the same loop runs them in
// sequence (deferred futures run when waited on). Waiting for the previous
// reconstruction keeps the blending order, and bounds retained ORT outputs to two.
std::vector<float> separate_at_model_rate(const ScnetModel& model, const float* audio, int channels,
                                          int64_t samples, Job& job) {
  job.throw_if_cancelled();
  if (samples == 0) return {};

  ModelInput input = normalized_stereo(audio, channels, samples);
  Blender blender(samples);
  RunScope run(job);
  const int64_t chunks = (samples + kStride - 1) / kStride;
  int64_t chunks_done = 0;  // reconstructions run one at a time, in order

  const auto prepare = [&](int64_t offset) {
    const int64_t length = std::min(kSegment, samples - offset);
    std::vector<float> chunk(kModelChannels * length);
    for (int c = 0; c < kModelChannels; ++c) {
      const float* source = input.stereo.data() + c * samples + offset;
      std::copy(source, source + length, chunk.begin() + c * length);
    }
    return model.prepare(chunk.data(), length);
  };

  // A cancelled job makes ORT abort the Run with an error: report it as a cancellation.
  const auto infer = [&](ScnetModel::Input& prepared) {
    job.throw_if_cancelled();
    try {
      return model.infer(prepared, run.options());
    } catch (const Ort::Exception&) {
      job.throw_if_cancelled();
      throw;
    }
  };

  // Reconstruction takes a while (~0.3 s per chunk on CPU): check before and after it.
  const auto reconstruct_and_blend = [&](ScnetModel::Output prediction, int64_t offset) {
    job.throw_if_cancelled();
    std::vector<float> chunk(kSignals * prediction.samples);
    model.reconstruct(prediction, chunk.data());
    job.throw_if_cancelled();
    blender.add(chunk, prediction.samples, offset);
    job.set_progress(static_cast<double>(++chunks_done) / chunks);
  };

  // Futures are declared after everything they capture, so they're joined first,
  // also when an exception leaves the loop.
  const auto policy = model.workers() > 1 ? std::launch::async : std::launch::deferred;
  auto prepared = std::async(policy, prepare, int64_t{0});
  std::future<void> reconstructed;
  for (int64_t offset = 0; offset < samples; offset += kStride) {
    auto chunk_input = prepared.get();
    if (offset + kStride < samples) prepared = std::async(policy, prepare, offset + kStride);
    auto prediction = infer(chunk_input);
    if (reconstructed.valid()) reconstructed.get();
    reconstructed = std::async(policy, reconstruct_and_blend, std::move(prediction), offset);
  }
  reconstructed.get();

  input.stereo = {};  // not needed anymore: free it before allocating the result
  std::vector<float> stems = blender.finish(input.norm);
  if (channels == 1) downmix_in_place(stems, samples);
  return stems;
}

// Resamples each of `signals` planar channels, one thread per channel.
std::vector<float> resample_planar(const Resampler& resampler, const float* audio, int signals,
                                   int64_t frames) {
  const int64_t out_frames = resampler.output_length(frames);
  std::vector<float> out(signals * out_frames);
  std::vector<std::future<void>> workers;
  workers.reserve(signals);
  for (int s = 0; s < signals; ++s) {
    workers.push_back(std::async(std::launch::async, [&, s] {
      const std::vector<float> resampled = resampler.run(audio + s * frames, frames);
      std::copy(resampled.begin(), resampled.end(), out.begin() + s * out_frames);
    }));
  }
  for (auto& worker : workers) worker.get();
  return out;
}

}  // namespace

bool samplerate_supported(int samplerate) {
  if (samplerate < kMinSamplerate || samplerate > kMaxSamplerate) return false;
  return resampling_practical(samplerate, kSamplerate) && resampling_practical(kSamplerate, samplerate);
}

// Same path as the original inference: to 44.1 kHz, separate, back to the input rate.
Separation separate_track(const ScnetModel& model, const float* audio, int channels, int64_t frames,
                          int samplerate, Job& job) {
  if (samplerate == kSamplerate) return {frames, separate_at_model_rate(model, audio, channels, frames, job)};

  const Resampler to_model(samplerate, kSamplerate);
  const Resampler back(kSamplerate, samplerate);
  const int64_t model_frames = to_model.output_length(frames);

  // Resampling takes ~0.3 s per channel for 4 minutes of audio and doesn't stop halfway:
  // checking after each pass is enough, and a job cancelled at the very end isn't a success.
  std::vector<float> input = resample_planar(to_model, audio, channels, frames);
  job.throw_if_cancelled();
  const std::vector<float> separated =
      separate_at_model_rate(model, input.data(), channels, model_frames, job);
  input = {};
  std::vector<float> stems = resample_planar(back, separated.data(), kStemCount * channels, model_frames);
  job.throw_if_cancelled();
  return {back.output_length(model_frames), std::move(stems)};
}

}  // namespace stems
