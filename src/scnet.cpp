#include "scnet.h"

#include <algorithm>
#include <future>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "spectrogram.h"

namespace stems {
namespace {

// Very short inputs are zero-padded up to this length. The original code crashes on
// them (reflect padding needs more samples than half a window), so there's no
// reference behavior to match; this just keeps the STFT valid.
constexpr int64_t kMinSamples = kFftSize;

// One ORT environment for the whole process (ORT wants exactly one, living longest).
Ort::Env& ort_env() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "stems");
  return env;
}

// Same padding as SCNet.forward(): up to a multiple of the hop, and such that the
// frame count before centering is odd (the network's internal rfft needs an even
// number of STFT frames). Note it always adds at least something, even when
// `samples` is already a multiple of the hop: we keep that quirk to stay identical.
int64_t model_padding(int64_t samples) {
  int64_t padding = kHop - samples % kHop;
  if ((samples + padding) / kHop % 2 == 0) padding += kHop;
  return padding;
}

// Runs the network on a GPU. TF32 stays off so results keep matching the FP32 original,
// and the arena grows by what's asked instead of doubling, to keep VRAM down.
// Fails with an Ort::Exception on a CPU-only build or without CUDA/cuDNN on the system.
void use_cuda(Ort::SessionOptions& options, int device) {
  Ort::CUDAProviderOptions cuda;
  cuda.Update({{"device_id", std::to_string(device)},
               {"use_tf32", "0"},
               {"cudnn_conv_algo_search", "HEURISTIC"},
               {"cudnn_conv_use_max_workspace", "0"}});
  options.AppendExecutionProvider_CUDA_V2(*cuda);
}

}  // namespace

ScnetModel::ScnetModel(const std::string& onnx_path, int threads, int cuda_device) {
  if (std::string(OrtGetApiBase()->GetVersionString()) != STEMS_ORT_VERSION) {
    throw Ort::Exception("ONNX Runtime version mismatch: expected " STEMS_ORT_VERSION, ORT_FAIL);
  }
  if (threads <= 0) threads = std::max(1u, std::thread::hardware_concurrency() / 2);
  fft_threads_ = std::min(threads, 4);

  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(threads);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // The memory pattern pre-plans one big block per input shape and keeps it. Our chunks
  // come in several lengths, so those blocks pile up: off, peak RAM per 20 s chunk drops
  // from ~9.4 to ~5.5 GiB at the same speed.
  options.DisableMemPattern();
  // Keep the CPU arena: without it a 25 s excerpt peaks ~300 MB lower, but a 4 min
  // track ~450 MB higher (malloc fragments over many chunks).

  Ort::Env& env = ort_env();  // before the CUDA options: ORT logs through it
  if (cuda_device >= 0) use_cuda(options, cuda_device);
  session_ = std::make_unique<Ort::Session>(env, onnx_path.c_str(), options);
}

ScnetModel::~ScnetModel() = default;

ScnetModel::Input ScnetModel::prepare(const float* audio, int64_t samples) const {
  // Pad each channel with zeros at the end, like F.pad(x, (0, padding)).
  const int64_t padded = std::max(samples + model_padding(samples), kMinSamples + model_padding(kMinSamples));
  const int64_t frames = stft_frames(padded);
  const int64_t plane = kBins * frames;

  std::vector<float> channel(padded);
  std::vector<float> spectrum(2 * kModelChannels * plane);  // [left re, left im, right re, right im]
  for (int c = 0; c < kModelChannels; ++c) {
    std::fill(channel.begin(), channel.end(), 0.0f);
    std::copy(audio + c * samples, audio + (c + 1) * samples, channel.begin());
    stft(channel.data(), padded, &spectrum[(2 * c) * plane], &spectrum[(2 * c + 1) * plane]);
  }
  return {samples, frames, std::move(spectrum)};
}

ScnetModel::Output ScnetModel::infer(Input& prepared, Ort::RunOptions& run_options) const {
  const int64_t frames = prepared.frames;
  // Run the network: (1, 4, bins, frames) -> (1, 16, bins, frames).
  const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::array<int64_t, 4> shape = {1, 2 * kModelChannels, kBins, frames};
  Ort::Value input = Ort::Value::CreateTensor<float>(memory, prepared.spectrum.data(),
                                                     prepared.spectrum.size(), shape.data(), shape.size());
  const char* input_names[] = {"spectrum"};
  const char* output_names[] = {"stems"};
  auto outputs = session_->Run(run_options, input_names, &input, 1, output_names, 1);
  if (outputs.size() != 1 || !outputs[0].IsTensor()) {
    throw Ort::Exception("incompatible model output: expected one stems tensor", ORT_INVALID_GRAPH);
  }
  const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
  if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw Ort::Exception("incompatible model output: stems must be float32", ORT_INVALID_GRAPH);
  }
  const std::vector<int64_t> expected_shape = {1, 2 * kStemCount * kModelChannels, kBins, frames};
  if (output_info.GetShape() != expected_shape) {
    throw Ort::Exception("incompatible model output: expected stems shape [1, 16, 2049, input_frames]",
                         ORT_INVALID_GRAPH);
  }
  return {prepared.samples, frames, std::move(outputs[0])};
}

void ScnetModel::reconstruct(const Output& output, float* out) const {
  const int64_t samples = output.samples;
  const int64_t frames = output.frames;
  const int64_t plane = kBins * frames;
  const float* result = output.spectrum.GetTensorData<float>();

  // Output channel 4 * stem + 2 * channel + (0 real, 1 imag). iSTFT each, keep the first `samples`.
  // Channels are independent: parallelize them, preserving each channel's FFT
  // and overlap-add order. At most four reconstruction workers, even on large CPUs.
  const auto reconstruct_channels = [&](int worker) {
    std::vector<float> signal(kHop * (frames - 1));
    for (int channel = worker; channel < kStemCount * kModelChannels; channel += fft_threads_) {
      const float* real = result + 2 * channel * plane;
      const float* imag = result + (2 * channel + 1) * plane;
      istft(real, imag, frames, signal.data());
      std::copy(signal.begin(), signal.begin() + samples, out + channel * samples);
    }
  };
  std::vector<std::future<void>> workers;
  workers.reserve(fft_threads_ - 1);
  for (int worker = 1; worker < fft_threads_; ++worker) {
    workers.push_back(std::async(std::launch::async, reconstruct_channels, worker));
  }
  reconstruct_channels(0);
  for (auto& worker : workers) worker.get();
}

}  // namespace stems
