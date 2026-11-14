#pragma once

#include <cstdint>

namespace stems {

// STFT/iSTFT exactly as SCNet calls torch.stft/torch.istft:
// n_fft 4096, hop 1024, no window (= rectangular), center=True with reflect padding,
// normalized=True (scaled by 1/sqrt(n_fft)).
inline constexpr int64_t kFftSize = 4096;
inline constexpr int64_t kHop = 1024;
inline constexpr int64_t kBins = kFftSize / 2 + 1;  // 2049

// Number of frames torch.stft produces for `samples` input samples (center=True).
inline int64_t stft_frames(int64_t samples) { return samples / kHop + 1; }

// Spectrum of one channel. `samples` must be a multiple of kHop and larger than kFftSize / 2
// (reflect padding needs that). Writes real and imaginary parts as [bin][frame] planes,
// each kBins * stft_frames(samples) floats: the layout the ONNX model expects.
void stft(const float* audio, int64_t samples, float* real, float* imag);

// Inverse of stft(): reads [bin][frame] planes with `frames` frames and writes
// kHop * (frames - 1) samples, like torch.istft without `length`.
void istft(const float* real, const float* imag, int64_t frames, float* audio);

}  // namespace stems
