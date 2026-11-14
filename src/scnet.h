#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace Ort {
struct Session;
}

namespace stems {

inline constexpr int kSamplerate = 44100;
inline constexpr int kModelChannels = 2;
inline constexpr int kStemCount = 4;
inline constexpr std::array<const char*, kStemCount> kStemNames = {"drums", "bass", "other", "vocals"};

// SCNet on one chunk of stereo audio: the native version of SCNet.forward().
// Loaded once, then read-only: separate() can run from several threads at once.
class ScnetModel {
 public:
  ScnetModel(const std::string& onnx_path, int threads);
  ~ScnetModel();

  // `audio`: 2 planar channels of `samples` each (already normalized by the caller).
  // `out`:   kStemCount * 2 * samples floats, [stem][channel][sample].
  void separate(const float* audio, int64_t samples, float* out) const;

 private:
  std::unique_ptr<Ort::Session> session_;
};

}  // namespace stems
