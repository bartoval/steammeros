#pragma once

#include <atomic>
#include <exception>
#include <mutex>

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
  // Also stops a model run in progress: ORT checks the terminate flag between graph
  // nodes, so we don't wait for the rest of a 20 s chunk.
  void cancel() {
    cancelled_ = true;
    std::lock_guard lock(mutex_);
    if (running_) running_->SetTerminate();
  }
  bool cancelled() const { return cancelled_; }
  void throw_if_cancelled() const {
    if (cancelled_) throw Cancelled();
  }

  double progress() const { return progress_; }
  void set_progress(double fraction) { progress_ = fraction; }

  // While a separation runs the model, it lends its run options here (see RunScope),
  // so cancel() can terminate them. Checking the flag under the lock means a cancel
  // can't slip in between.
  void attach(Ort::RunOptions* options) {
    std::lock_guard lock(mutex_);
    running_ = options;
    if (cancelled_ && running_) running_->SetTerminate();
  }
  void detach() noexcept {
    std::lock_guard lock(mutex_);
    running_ = nullptr;
  }

 private:
  std::atomic<bool> cancelled_{false};
  std::atomic<double> progress_{0.0};
  std::mutex mutex_;                    // guards running_
  Ort::RunOptions* running_ = nullptr;  // owned by the separation, not the job
};

// Run options for one separation, attached to its job for as long as this lives.
class RunScope {
 public:
  explicit RunScope(Job& job) : job_(job) { job_.attach(&options_); }
  ~RunScope() { job_.detach(); }
  RunScope(const RunScope&) = delete;
  RunScope& operator=(const RunScope&) = delete;

  Ort::RunOptions& options() { return options_; }

 private:
  Job& job_;
  Ort::RunOptions options_;
};

}  // namespace stems
