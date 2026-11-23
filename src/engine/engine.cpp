#include "engine/engine.h"

#include <memory>
#include <mutex>

#include "engine/lifecycle.h"
#include "inference/scheduler.hpp"
#include "inference/worker.hpp"
#include "logging/log.h"
#include "worker/worker.hpp"

namespace zinferlm {
namespace internal {

Engine &Engine::instance() {
  static Engine engine;
  return engine;
}

Engine::~Engine() { shutdown(); }

bool Engine::running() const {
  return running_.load(std::memory_order_acquire);
}

void Engine::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (shutting_down_.load(std::memory_order_acquire)) {
    throw worker_stopped("engine is shutting down");
  }
  if (started_.load(std::memory_order_acquire)) {
    return;
  }
  // Constructed inside a member function so the private constructors stay
  // inaccessible to everyone else (std::make_shared would not be a friend).
  auto scheduler = std::shared_ptr<InferenceScheduler>(new InferenceScheduler());
  auto worker = std::shared_ptr<InferenceWorker>(new InferenceWorker());
  scheduler_ = std::move(scheduler);
  worker_ = std::move(worker);
  started_.store(true, std::memory_order_release);
  running_.store(true, std::memory_order_release);
  engine_mark_started();
  LOG_INF("engine started\n");
}

std::shared_ptr<InferenceWorker> Engine::worker() {
  start();
  std::lock_guard<std::mutex> lock(mutex_);
  return worker_;
}

std::shared_ptr<InferenceScheduler> Engine::scheduler() {
  start();
  std::lock_guard<std::mutex> lock(mutex_);
  return scheduler_;
}

void Engine::shutdown() {
  if (shutting_down_.exchange(true, std::memory_order_acq_rel)) {
    return; // already shut down, or currently shutting down elsewhere
  }

  std::shared_ptr<InferenceWorker> worker;
  std::shared_ptr<InferenceScheduler> scheduler;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    worker = worker_;
    scheduler = scheduler_;
  }

  // Joining from one of our own threads would deadlock. Refuse and leave the
  // liveness flag set so Model::unload() keeps refusing as well.
  if ((worker && worker->is_worker_thread()) ||
      (scheduler && scheduler->on_scheduler_thread())) {
    LOG_ERR("engine shutdown requested from an engine thread; ignoring\n");
    return;
  }

  if (worker) {
    worker->stop();
  }
  if (scheduler) {
    scheduler->stop();
    scheduler->release_state();
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    worker_.reset();
    scheduler_.reset();
  }

  running_.store(false, std::memory_order_release);
  engine_mark_stopped();
  LOG_INF("engine stopped\n");
}

} // namespace internal
} // namespace zinferlm
