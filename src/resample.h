#pragma once

#include <cstdint>
#include <vector>

namespace stems {

// Whether a track at `samplerate` can go to the model's 44.1 kHz and back.
// Rejects rates whose ratio to 44100 doesn't simplify (e.g. 44056 Hz): their
// filter table would take gigabytes.
bool samplerate_supported(int samplerate);

// Port of julius.resample_frac (the resampler the original SCNet inference uses):
// windowed-sinc polyphase filter, 24 zero crossings, cutoff at 94.5% of the lower Nyquist.
// One filter per output phase, built once; run() is read-only and thread-safe.
class Resampler {
 public:
  Resampler(int from_rate, int to_rate);

  // Same length rule as julius: floor(length * to / from).
  int64_t output_length(int64_t length) const;
  std::vector<float> run(const float* signal, int64_t length) const;

 private:
  // Taps of one output phase. Only [first, last) is non-zero: julius keeps the
  // rest too, but those taps are ~1e-20, so skipping them changes nothing audible
  // and cuts the work about 4x.
  struct Phase {
    std::vector<float> taps;
    int64_t first = 0;
    int64_t last = 0;
  };

  int64_t from_ = 1;  // rates divided by their gcd, e.g. 48000 -> 44100 is 160 -> 147
  int64_t to_ = 1;
  int64_t width_ = 0;  // samples of context on each side
  std::vector<Phase> phases_;
};

}  // namespace stems
