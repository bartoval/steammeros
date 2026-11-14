#include "resample.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "scnet.h"

namespace stems {
namespace {

constexpr int kZeros = 24;            // julius default
constexpr double kRolloff = 0.945;    // julius default
constexpr int kMinSamplerate = 8000;
constexpr int kMaxSamplerate = 384000;
constexpr int64_t kMaxTableSize = int64_t{1} << 22;  // floats, 16 MiB

constexpr double kPi = 3.14159265358979323846;

int64_t reduced(int64_t rate, int64_t other) { return rate / std::gcd(rate, other); }

// Context on each side of the filter, as julius computes it.
int64_t filter_width(int64_t from, int64_t to) {
  return static_cast<int64_t>(std::ceil(kZeros * from / (std::min(from, to) * kRolloff)));
}

int64_t table_size(int from_rate, int to_rate) {
  const int64_t from = reduced(from_rate, to_rate);
  const int64_t to = reduced(to_rate, from_rate);
  return to * (2 * filter_width(from, to) + from);
}

}  // namespace

bool samplerate_supported(int samplerate) {
  if (samplerate < kMinSamplerate || samplerate > kMaxSamplerate) return false;
  return table_size(samplerate, kSamplerate) <= kMaxTableSize &&
         table_size(kSamplerate, samplerate) <= kMaxTableSize;
}

// Builds the filter for each output phase: a sinc at the right sub-sample offset,
// shaped by a Hann-like window and normalized so a constant signal stays constant.
Resampler::Resampler(int from_rate, int to_rate)
    : from_(reduced(from_rate, to_rate)), to_(reduced(to_rate, from_rate)) {
  if (from_ == to_) return;
  width_ = filter_width(from_, to_);
  const double cutoff = std::min(from_, to_) * kRolloff;
  const int64_t size = 2 * width_ + from_;

  phases_.resize(to_);
  for (int64_t phase = 0; phase < to_; ++phase) {
    Phase& filter = phases_[phase];
    filter.taps.resize(size);
    filter.first = size;
    double sum = 0.0;
    for (int64_t k = 0; k < size; ++k) {
      const double t = (static_cast<double>(k - width_) / from_ - static_cast<double>(phase) / to_) * cutoff;
      if (std::abs(t) >= kZeros) continue;  // window is zero out there
      const double x = t * kPi;
      const double sinc = x == 0.0 ? 1.0 : std::sin(x) / x;
      const double window = std::pow(std::cos(x / kZeros / 2), 2);
      filter.taps[k] = static_cast<float>(sinc * window);
      sum += sinc * window;
      filter.first = std::min(filter.first, k);
      filter.last = k + 1;
    }
    for (float& tap : filter.taps) tap = static_cast<float>(tap / sum);
  }
}

// julius computes floor(to * length / from) on a float32 tensor. Above 2^24 samples
// that float rounds, so we round the same way to get the exact same length.
int64_t Resampler::output_length(int64_t length) const {
  const double exact = static_cast<double>(to_) * static_cast<double>(length) / static_cast<double>(from_);
  return static_cast<int64_t>(std::floor(static_cast<float>(exact)));
}

// Output sample `block * to + phase` is the dot product of phase's filter with the
// input around `block * from`. Edges repeat the first/last sample (replicate padding).
std::vector<float> Resampler::run(const float* signal, int64_t length) const {
  if (from_ == to_) return std::vector<float>(signal, signal + length);
  const int64_t count = output_length(length);
  std::vector<float> out(count);
  if (count == 0) return out;

  std::vector<float> padded(length + 2 * width_ + from_);
  std::fill(padded.begin(), padded.begin() + width_, signal[0]);
  std::copy(signal, signal + length, padded.begin() + width_);
  std::fill(padded.begin() + width_ + length, padded.end(), signal[length - 1]);

  for (int64_t i = 0; i < count; ++i) {
    const Phase& filter = phases_[i % to_];
    const float* window = padded.data() + (i / to_) * from_;
    out[i] = std::inner_product(filter.taps.begin() + filter.first, filter.taps.begin() + filter.last,
                                window + filter.first, 0.0f);
  }
  return out;
}

}  // namespace stems
