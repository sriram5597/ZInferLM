#pragma once

#include <atomic>
#include <memory>
#include <mutex>

namespace zinferlm {
namespace internal {

class InferenceScheduler;
class InferenceWorker;

// Owns the inference engine lifetime: the worker pool, the scheduler and every
// resource derived from the model. Internal only; the public shutdown surface
// is Chat::stop().
//
// Shutdown order is fixed: workers are stopped first (they are the only
// producers of sequences), then the scheduler thread is joined and its state
// released. Only after that is the model safe to unload.
class Engine {
public:
  static Engine &instance();

  // Starts the engine on first use. Throws worker_stopped once shutdown began.
  void start();

  // Cancels in-flight work, joins every engine thread and releases all state
  // derived from the model. Idempotent and safe to call from any non-engine
  // thread.
  void shutdown();

  bool running() const;

  // Shared ownership so a concurrent shutdown cannot free the object while a
  // caller is still submitting into it. Both accessors start the engine.
  std::shared_ptr<InferenceWorker> worker();
  std::shared_ptr<InferenceScheduler> scheduler();

  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;
  ~Engine();

private:
  Engine() = default;

  mutable std::mutex mutex_;
  std::shared_ptr<InferenceWorker> worker_;
  std::shared_ptr<InferenceScheduler> scheduler_;
  std::atomic<bool> started_{false};
  std::atomic<bool> running_{false};
  std::atomic<bool> shutting_down_{false};
};

} // namespace internal
} // namespace zinferlm
