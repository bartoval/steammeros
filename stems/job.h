#pragma once

#include <atomic>
#include <exception>

#include "onnxruntime_cxx_api.h"

namespace stems {

// Thrown inside a separation once its job has been cancelled.
struct Cancelled : std::exception {
  const char* what() const noexcept override { return "cancelled"; }
};

// Lets another thread follow and stop one running separation.
// cancel() and progress() are safe to call from any thread at any time.
class Job {
 public:
  // Also stops the model mid-chunk: ORT checks the terminate flag between graph
  // nodes, so we don't wait for the rest of a 20 s chunk.
  void cancel() {
    cancelled_ = true;
    run_options_.SetTerminate();
  }
  bool cancelled() const { return cancelled_; }
  void throw_if_cancelled() const {
    if (cancelled_) throw Cancelled();
  }

  double progress() const { return progress_; }
  void set_progress(double fraction) { progress_ = fraction; }

  Ort::RunOptions& run_options() { return run_options_; }

 private:
  std::atomic<bool> cancelled_{false};
  std::atomic<double> progress_{0.0};
  Ort::RunOptions run_options_;
};

}  // namespace stems
