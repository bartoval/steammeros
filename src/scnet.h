#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace stems {

inline constexpr int kSamplerate = 44100;
inline constexpr int kModelChannels = 2;
inline constexpr int kStemCount = 4;
inline constexpr std::array<const char*, kStemCount> kStemNames = {"drums", "bass", "other", "vocals"};

// SCNet on one chunk of stereo audio: the native version of SCNet.forward().
// Three stages, so the caller can overlap them across chunks: prepare() (STFT),
// infer() (the network), reconstruct() (iSTFT). Loaded once, then read-only:
// every stage can run from several threads at once.
class ScnetModel {
 public:
  // Owning buffers passed between the stages. Keeping the ORT output alive lets
  // reconstruction overlap the next Run without a copy.
  struct Input {
    int64_t samples;
    int64_t frames;
    std::vector<float> spectrum;
  };
  struct Output {
    int64_t samples;
    int64_t frames;
    Ort::Value spectrum;
  };

  // `cuda_device` >= 0 runs the network on that GPU (STFT and the rest stay on the CPU);
  // -1 runs everything on the CPU.
  ScnetModel(const std::string& onnx_path, int threads, int cuda_device);
  ~ScnetModel();

  // `audio`: 2 planar channels of `samples` each (already normalized by the caller).
  Input prepare(const float* audio, int64_t samples) const;
  // `run_options` can stop a running call from another thread (see Job).
  Output infer(Input& input, Ort::RunOptions& run_options) const;
  // `out`: kStemCount * 2 * samples floats, [stem][channel][sample].
  void reconstruct(const Output& output, float* out) const;
  bool pipeline_enabled() const { return fft_threads_ > 1; }

 private:
  std::unique_ptr<Ort::Session> session_;
  int fft_threads_ = 1;
};

}  // namespace stems
