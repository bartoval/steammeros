#include "spectrogram.h"

#include <algorithm>
#include <complex>
#include <functional>
#include <vector>

#include "pocketfft_hdronly.h"

namespace stems {
namespace {

using Complex = std::complex<float>;

constexpr int64_t kPad = kFftSize / 2;  // center=True pads half a window on each side

// irfft divides by n_fft; normalized=True scales both ways by 1/sqrt(n_fft) = 1/64.
constexpr float kScale = 1.0f / 64.0f;
static_assert(kFftSize == 64 * 64);

// torch "reflect" padding: index -1 maps to 1, n maps to n - 2 (the edge sample isn't repeated).
int64_t reflect(int64_t i, int64_t n) {
  if (i < 0) return -i;
  if (i >= n) return 2 * (n - 1) - i;
  return i;
}

// pocketfft sees the frames as a 2-D array [frame][sample in frame]. These strides
// (in bytes, one per axis) let it read and write our layouts directly, no copies:
// frames overlapping by a hop in the signal, spectra as [bin][frame].
pocketfft::shape_t frame_shape(int64_t frames) {
  return {static_cast<size_t>(frames), static_cast<size_t>(kFftSize)};
}
pocketfft::stride_t signal_strides() { return {sizeof(float) * kHop, sizeof(float)}; }
pocketfft::stride_t row_strides() { return {sizeof(float) * kFftSize, sizeof(float)}; }
pocketfft::stride_t spectrum_strides(int64_t frames) {
  return {sizeof(Complex), static_cast<ptrdiff_t>(sizeof(Complex) * frames)};
}

// How many frames overlap sample `i` of the padded signal: 4 in the middle,
// fewer at the edges. torch divides by this after overlap-add (rectangular window).
int64_t frames_covering(int64_t i, int64_t frames) {
  const int64_t first = i < kFftSize ? 0 : (i - kFftSize) / kHop + 1;
  const int64_t last = std::min(i / kHop, frames - 1);
  return last - first + 1;
}

}  // namespace

void stft(const float* audio, int64_t samples, float* real, float* imag) {
  const int64_t frames = stft_frames(samples);

  // Reflect-pad, then FFT every frame straight out of the padded signal.
  std::vector<float> padded(samples + 2 * kPad);
  for (int64_t i = 0; i < static_cast<int64_t>(padded.size()); ++i) {
    padded[i] = audio[reflect(i - kPad, samples)];
  }
  std::vector<Complex> spectrum(kBins * frames);
  pocketfft::r2c(frame_shape(frames), signal_strides(), spectrum_strides(frames), /*axis=*/1,
                 pocketfft::FORWARD, padded.data(), spectrum.data(), kScale);

  for (int64_t i = 0; i < kBins * frames; ++i) {
    real[i] = spectrum[i].real();
    imag[i] = spectrum[i].imag();
  }
}

void istft(const float* real, const float* imag, int64_t frames, float* audio) {
  std::vector<Complex> spectrum(kBins * frames);
  for (int64_t i = 0; i < kBins * frames; ++i) spectrum[i] = {real[i], imag[i]};

  std::vector<float> rows(frames * kFftSize);
  pocketfft::c2r(frame_shape(frames), spectrum_strides(frames), row_strides(), /*axis=*/1,
                 pocketfft::BACKWARD, spectrum.data(), rows.data(), kScale);

  // Overlap-add the frames back into one signal.
  std::vector<float> sum(kFftSize + kHop * (frames - 1), 0.0f);
  for (int64_t f = 0; f < frames; ++f) {
    const float* row = rows.data() + f * kFftSize;
    float* target = sum.data() + f * kHop;
    std::transform(row, row + kFftSize, target, target, std::plus<>());
  }

  // Normalize by frame coverage and drop the center=True padding.
  const int64_t samples = kHop * (frames - 1);
  for (int64_t i = 0; i < samples; ++i) {
    audio[i] = sum[i + kPad] / static_cast<float>(frames_covering(i + kPad, frames));
  }
}

}  // namespace stems
