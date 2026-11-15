// Optional ONNX Runtime operator for the CUDA model (tools/export_cuda_onnx.py): a
// bidirectional LSTM with FP16 weights and inputs computed in FP32 by cuDNN. ONNX
// Runtime's own FP16 LSTM also computes in FP16, which costs too much precision.
// Built separately (tools/build_cuda_ops.sh): the Go package needs no CUDA.
#define ORT_API_MANUAL_INIT
#include <cuda_runtime_api.h>
#include <cudnn.h>

#include <array>
#include <climits>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace {

constexpr auto kType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
constexpr size_t kScalar = 2;  // bytes per FP16 value

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check(cudnnStatus_t status) {
  if (status != CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(status));
}

// Exceptions must not cross into ONNX Runtime: turn them into a status.
template <class F>
OrtStatusPtr guarded(F&& fn) noexcept {
  try {
    fn();
    return nullptr;
  } catch (const Ort::Exception& e) {
    return Ort::GetApi().CreateStatus(e.GetOrtErrorCode(), e.what());
  } catch (const std::exception& e) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, "unknown cuDNN LSTM error");
  }
}

// Selects a CUDA device for the scope, restoring the previous one.
struct Device {
  int previous = 0;
  explicit Device(int device) {
    check(cudaGetDevice(&previous));
    if (previous != device) check(cudaSetDevice(device));
  }
  ~Device() { cudaSetDevice(previous); }
};

// Temporary device memory from ORT's arena, so concurrent Runs never share a buffer.
struct Scratch {
  Ort::Allocator allocator;
  void* ptr = nullptr;
  Scratch(Ort::KernelContext& context, const Ort::MemoryInfo& memory, size_t bytes)
      : allocator(context.GetAllocator(*static_cast<const OrtMemoryInfo*>(memory))) {
    if (!bytes) return;
    OrtSyncStream* stream = nullptr;
    Ort::ThrowOnError(Ort::GetApi().KernelContext_GetSyncStream(context.GetOrtKernelContext(), &stream));
    auto* raw = static_cast<OrtAllocator*>(allocator);
    if (!raw->AllocOnStream) throw std::runtime_error("cuDNN LSTM requires a stream-aware allocator");
    // ORT 1.31's GetScratchBuffer can terminate on allocation failure. Calling
    // the allocator directly lets ComputeV2 return that failure as an ORT status.
    ptr = raw->AllocOnStream(raw, bytes, stream);
    if (!ptr) throw std::bad_alloc();
  }
  ~Scratch() {
    if (ptr) allocator.Free(ptr);
  }
};

// cuDNN objects and packed weights, created on the first Run.
struct State {
  int device = 0;
  cudnnHandle_t handle = nullptr;
  cudnnDropoutDescriptor_t dropout = nullptr;
  cudnnRNNDescriptor_t rnn = nullptr;
  cudnnRNNDataDescriptor_t xdesc = nullptr, ydesc = nullptr;
  cudnnTensorDescriptor_t hidden_desc = nullptr, matrix = nullptr, bias = nullptr;
  void* weights = nullptr;
  void* dropout_state = nullptr;
  size_t weight_bytes = 0;
  int steps = 0, batch = 0;
  size_t workspace_bytes = 0, reserve_bytes = 0;

  ~State() {
    int previous = device;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (matrix) cudnnDestroyTensorDescriptor(matrix);
    if (bias) cudnnDestroyTensorDescriptor(bias);
    if (hidden_desc) cudnnDestroyTensorDescriptor(hidden_desc);
    if (xdesc) cudnnDestroyRNNDataDescriptor(xdesc);
    if (ydesc) cudnnDestroyRNNDataDescriptor(ydesc);
    if (rnn) cudnnDestroyRNNDescriptor(rnn);
    if (dropout) cudnnDestroyDropoutDescriptor(dropout);
    if (handle) cudnnDestroy(handle);
    if (weights) cudaFree(weights);
    if (dropout_state) cudaFree(dropout_state);
    cudaSetDevice(previous);
  }
};

// Element count of a dense cuDNN parameter tensor; rejects any other layout.
size_t elements(cudnnTensorDescriptor_t desc) {
  cudnnDataType_t dtype;
  int rank = 0, dims[8], strides[8];
  check(cudnnGetTensorNdDescriptor(desc, 8, &dtype, &rank, dims, strides));
  size_t n = 1;
  for (int i = rank - 1; i >= 0; --i) {
    if (dims[i] > 1 && static_cast<size_t>(strides[i]) != n) {
      throw std::runtime_error("unsupported cuDNN parameter layout");
    }
    n *= dims[i];
  }
  return n;
}

// Inputs: X [sequence,batch,hidden], then the ONNX LSTM's W, R, B as constants.
// Output: [sequence,batch,direction,hidden], the layout SCNet transposes ORT's LSTM into.
struct Kernel {
  int hidden = 0;
  std::array<std::vector<char>, 3> source;  // W, R, B until the first Run copies them
  std::unique_ptr<State> state;
  std::mutex mutex;

  explicit Kernel(const OrtKernelInfo* raw) {
    Ort::ConstKernelInfo info(raw);
    const int64_t h = info.GetAttribute<int64_t>("hidden_size");
    if (h <= 0 || h > INT_MAX / 8) throw std::runtime_error("invalid LSTM hidden size");
    hidden = static_cast<int>(h);
    for (int i = 0; i < 3; ++i) {
      int constant = 0;
      const auto value = info.GetTensorConstantInput(i + 1, &constant);
      if (!constant || !value) throw std::runtime_error("cuDNN LSTM weights must be constant initializers");
      const auto tensor = value.GetTensorTypeAndShapeInfo();
      const std::vector<int64_t> expected =
          i < 2 ? std::vector<int64_t>{2, 4 * h, h} : std::vector<int64_t>{2, 8 * h};
      if (tensor.GetElementType() != kType || tensor.GetShape() != expected) {
        throw std::runtime_error("invalid cuDNN LSTM weight shape or type");
      }
      source[i].resize(tensor.GetElementCount() * kScalar);
      std::memcpy(source[i].data(), value.GetTensorRawData(), source[i].size());
    }
  }

  void initialize(int device, cudaStream_t stream) {
    auto s = std::make_unique<State>();
    s->device = device;
    check(cudnnCreate(&s->handle));
    check(cudnnSetStream(s->handle, stream));
    check(cudnnCreateDropoutDescriptor(&s->dropout));
    size_t bytes = 0;
    check(cudnnDropoutGetStatesSize(s->handle, &bytes));
    if (bytes) check(cudaMalloc(&s->dropout_state, bytes));
    check(cudnnSetDropoutDescriptor(s->dropout, s->handle, 0, s->dropout_state, bytes, 0));
    check(cudnnCreateRNNDescriptor(&s->rnn));
    // FP16 data, FP32 math precision: as PyTorch runs an FP16 LSTM on cuDNN.
    check(cudnnSetRNNDescriptor_v8(s->rnn, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM, CUDNN_RNN_DOUBLE_BIAS,
                                   CUDNN_BIDIRECTIONAL, CUDNN_LINEAR_INPUT, CUDNN_DATA_HALF, CUDNN_DATA_FLOAT,
                                   CUDNN_TENSOR_OP_MATH, hidden, hidden, hidden, 1, s->dropout,
                                   CUDNN_RNN_PADDED_IO_ENABLED));
    check(cudnnCreateRNNDataDescriptor(&s->xdesc));
    check(cudnnCreateRNNDataDescriptor(&s->ydesc));
    check(cudnnCreateTensorDescriptor(&s->hidden_desc));
    check(cudnnCreateTensorDescriptor(&s->matrix));
    check(cudnnCreateTensorDescriptor(&s->bias));
    check(cudnnGetRNNWeightSpaceSize(s->handle, s->rnn, &s->weight_bytes));
    check(cudaMalloc(&s->weights, s->weight_bytes));
    check(cudaMemsetAsync(s->weights, 0, s->weight_bytes, stream));

    // Copy each gate's matrices and biases where cuDNN wants them.
    constexpr int kGateOrder[]{0, 2, 3, 1};  // cuDNN i,f,c,o -> ONNX i,o,f,c
    const size_t matrix_size = static_cast<size_t>(hidden) * hidden;
    for (int direction = 0; direction < 2; ++direction) {
      for (int layer = 0; layer < 8; ++layer) {  // 0-3 input weights, 4-7 recurrent
        void *w = nullptr, *b = nullptr;
        check(cudnnGetRNNWeightParams(s->handle, s->rnn, direction, s->weight_bytes, s->weights, layer,
                                      s->matrix, &w, s->bias, &b));
        if (!w || !b || elements(s->matrix) != matrix_size ||
            elements(s->bias) != static_cast<size_t>(hidden)) {
          throw std::runtime_error("unexpected cuDNN parameter shape");
        }
        const int kind = layer / 4, gate = kGateOrder[layer % 4];
        const char* sw = source[kind].data() + (direction * 4 + gate) * matrix_size * kScalar;
        const char* sb =
            source[2].data() + (direction * 8 + kind * 4 + gate) * static_cast<size_t>(hidden) * kScalar;
        check(cudaMemcpyAsync(w, sw, matrix_size * kScalar, cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(b, sb, static_cast<size_t>(hidden) * kScalar, cudaMemcpyHostToDevice, stream));
      }
    }
    // Once per kernel: the weights must be on the device before the host copies go.
    check(cudaStreamSynchronize(stream));
    state = std::move(s);
    for (auto& data : source) std::vector<char>().swap(data);
  }

  OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
    return guarded([&] {
      Ort::KernelContext context(raw);
      const auto x = context.GetInput(0);
      const auto shape = x.GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() != 3 || shape[0] <= 0 || shape[0] > INT_MAX || shape[1] <= 0 ||
          shape[1] > INT_MAX / hidden || shape[2] != hidden) {
        throw std::runtime_error("cuDNN LSTM expects [sequence,batch,hidden] with positive dimensions");
      }
      const int steps = static_cast<int>(shape[0]), batch = static_cast<int>(shape[1]);
      const int device = x.GetTensorMemoryInfo().GetDeviceId();
      Device selected(device);
      const auto stream = static_cast<cudaStream_t>(context.GetGPUComputeStream());
      // Guards the host-side descriptors only: the work itself runs asynchronously on
      // ORT's stream, with scratch memory of its own.
      std::lock_guard lock(mutex);
      if (!state) initialize(device, stream);
      auto& s = *state;
      if (s.device != device) throw std::runtime_error("LSTM kernel cannot move between CUDA devices");
      check(cudnnSetStream(s.handle, stream));
      if (s.steps != steps || s.batch != batch) {
        const std::vector<int> lengths(batch, steps);
        float padding = 0;
        check(cudnnSetRNNDataDescriptor(s.xdesc, CUDNN_DATA_HALF, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                                        steps, batch, hidden, lengths.data(), &padding));
        check(cudnnSetRNNDataDescriptor(s.ydesc, CUDNN_DATA_HALF, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                                        steps, batch, 2 * hidden, lengths.data(), &padding));
        const int dims[]{2, batch, hidden}, strides[]{batch * hidden, hidden, 1};
        check(cudnnSetTensorNdDescriptor(s.hidden_desc, CUDNN_DATA_HALF, 3, dims, strides));
        check(cudnnGetRNNTempSpaceSizes(s.handle, s.rnn, CUDNN_FWD_MODE_INFERENCE, s.xdesc,
                                        &s.workspace_bytes, &s.reserve_bytes));
        s.steps = steps;
        s.batch = batch;
      }
      const Ort::MemoryInfo memory("Cuda", OrtDeviceAllocator, device, OrtMemTypeDefault);
      Scratch work(context, memory, s.workspace_bytes);
      Scratch reserve(context, memory, s.reserve_bytes);
      auto y = context.GetOutput(0, {shape[0], shape[1], 2, hidden});
      check(cudnnRNNForward(s.handle, s.rnn, CUDNN_FWD_MODE_INFERENCE, nullptr, s.xdesc, x.GetTensorRawData(),
                            s.ydesc, y.GetTensorMutableRawData(), s.hidden_desc, nullptr, nullptr,
                            s.hidden_desc, nullptr, nullptr, s.weight_bytes, s.weights, s.workspace_bytes,
                            work.ptr, s.reserve_bytes, reserve.ptr));
    });
  }
};

struct PackedLSTM16 : Ort::CustomOpBase<PackedLSTM16, Kernel, true> {
  const char* GetName() const { return "PackedLSTM16"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 4; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return kType; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return kType; }
  OrtMemType GetInputMemoryType(size_t index) const {
    return index == 0 ? OrtMemTypeDefault : OrtMemTypeCPUInput;
  }
  OrtStatusPtr CreateKernelV2(const OrtApi&, const OrtKernelInfo* info, void** out) const noexcept {
    *out = nullptr;
    return guarded([&] { *out = new Kernel(info); });
  }
};

}  // namespace

// Entry point ORT looks up when the session options register this library.
extern "C" __attribute__((visibility("default"))) OrtStatusPtr ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(ORT_API_VERSION);
  if (!api)
    return base->GetApi(1)->CreateStatus(ORT_FAIL, "CUDA operators require ONNX Runtime 1.31 or newer");
  static std::once_flag init;
  std::call_once(init, [&] { Ort::InitApi(api); });
  return guarded([&] {
    // The domain must outlive every session using it: keep one for the process.
    static PackedLSTM16 op;
    static Ort::CustomOpDomain domain("com.steammeros.cuda");
    static std::once_flag added;
    std::call_once(added, [&] { domain.Add(&op); });
    Ort::UnownedSessionOptions(options).Add(domain);
  });
}
