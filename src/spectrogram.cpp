#include "spectrogram.h"

#include <cmath>
#include <complex>
#include <vector>

#include "pocketfft_hdronly.h"

namespace stems {
namespace {

constexpr int64_t kPad = kFftSize / 2;  // center=True pads half a window on each side

// torch "reflect" padding: index -1 maps to 1, n maps to n - 2 (the edge sample isn't repeated).
int64_t reflect(int64_t i, int64_t n) {
  if (i < 0) return -i;
  if (i >= n) return 2 * (n - 1) - i;
  return i;
}

pocketfft::shape_t frame_shape(int64_t frames) {
  return {static_cast<size_t>(frames), static_cast<size_t>(kFftSize)};
}

}  // namespace

void stft(const float* audio, int64_t samples, float* real, float* imag) {
  const int64_t frames = stft_frames(samples);

  // Cut the padded signal into overlapping frames, one row each, then FFT all rows at once.
  std::vector<float> rows(frames * kFftSize);
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t k = 0; k < kFftSize; ++k) {
      rows[f * kFftSize + k] = audio[reflect(f * kHop + k - kPad, samples)];
    }
  }

  std::vector<std::complex<float>> spectrum(frames * kBins);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kFftSize));  // normalized=True
  pocketfft::r2c(frame_shape(frames),
                 {sizeof(float) * kFftSize, sizeof(float)},
                 {sizeof(std::complex<float>) * kBins, sizeof(std::complex<float>)},
                 /*axis=*/1, pocketfft::FORWARD, rows.data(), spectrum.data(), scale);

  // [frame][bin] complex -> [bin][frame] real/imag planes.
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t b = 0; b < kBins; ++b) {
      real[b * frames + f] = spectrum[f * kBins + b].real();
      imag[b * frames + f] = spectrum[f * kBins + b].imag();
    }
  }
}

void istft(const float* real, const float* imag, int64_t frames, float* audio) {
  std::vector<std::complex<float>> spectrum(frames * kBins);
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t b = 0; b < kBins; ++b) {
      spectrum[f * kBins + b] = {real[b * frames + f], imag[b * frames + f]};
    }
  }

  // irfft divides by n_fft; normalized=True multiplies back by sqrt(n_fft).
  std::vector<float> rows(frames * kFftSize);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kFftSize));
  pocketfft::c2r(frame_shape(frames),
                 {sizeof(std::complex<float>) * kBins, sizeof(std::complex<float>)},
                 {sizeof(float) * kFftSize, sizeof(float)},
                 /*axis=*/1, pocketfft::BACKWARD, spectrum.data(), rows.data(), scale);

  // Overlap-add. With a rectangular window, torch divides each sample by how many
  // frames cover it (4 in the middle, fewer near the edges).
  const int64_t padded = kFftSize + kHop * (frames - 1);
  std::vector<float> sum(padded, 0.0f);
  std::vector<float> count(padded, 0.0f);
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t k = 0; k < kFftSize; ++k) {
      sum[f * kHop + k] += rows[f * kFftSize + k];
      count[f * kHop + k] += 1.0f;
    }
  }

  // Drop the center=True padding again.
  const int64_t samples = kHop * (frames - 1);
  for (int64_t i = 0; i < samples; ++i) {
    audio[i] = sum[i + kPad] / count[i + kPad];
  }
}

}  // namespace stems
