#pragma once

#include <condition_variable>
#include <exception>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace zinferlm {
namespace internal {

// Thrown when a task is submitted to a stopped worker, or when a queued task is
// cancelled during shutdown. Delivered either synchronously from add_task(), or
// asynchronously through the task's std::future.
class worker_stopped : public std::runtime_error {
public:
  explicit worker_stopped(const std::string &msg) : std::runtime_error(msg) {}
};

template <typename T, typename R> class Worker {
protected:
  int max_workers_ = 1;
  std::stop_source stop_source_;
  std::vector<std::jthread> worker_threads_;
  std::vector<std::thread::id> worker_thread_ids_;
  std::condition_variable_any cv_;
  std::mutex task_lock_;
  std::queue<std::pair<T, std::promise<R>>> tasks_;
  virtual void execute_task_() = 0;
  Worker(int max_workers);

public:
  ~Worker();
  void run(std::stop_token tok);
  void add_task(T &task, std::promise<R> result);
  void stop();

  // True when the calling thread is one of this pool's workers. Used by the
  // engine to refuse a shutdown request that would join itself.
  bool is_worker_thread() const;
};

template <typename T, typename R>
Worker<T, R>::Worker(int workers) : max_workers_(workers) {
  worker_threads_.reserve(max_workers_);
  worker_thread_ids_.reserve(max_workers_);
  std::stop_token tok = stop_source_.get_token();
  for (int i = 0; i < max_workers_; i++) {
    worker_threads_.emplace_back(
        std::jthread([this, tok]() { this->run(tok); }));
    worker_thread_ids_.push_back(worker_threads_.back().get_id());
  }
}

template <typename T, typename R> bool Worker<T, R>::is_worker_thread() const {
  const std::thread::id self = std::this_thread::get_id();
  for (const auto &id : worker_thread_ids_) {
    if (id == self) {
      return true;
    }
  }
  return false;
}

template <typename T, typename R> void Worker<T, R>::run(std::stop_token tok) {
  while (true) {
    std::unique_lock<std::mutex> lock(task_lock_);
    cv_.wait(lock, tok, [this] { return !tasks_.empty(); });

    if (tok.stop_requested()) {
      // Shutdown: cancel every queued task so no caller's future.get() hangs.
      while (!tasks_.empty()) {
        tasks_.front().second.set_exception(
            std::make_exception_ptr(worker_stopped("worker stopped")));
        tasks_.pop();
      }
      break;
    }
    lock.unlock();
    execute_task_();
  }
}

template <typename T, typename R>
void Worker<T, R>::add_task(T &task, std::promise<R> result) {
  {
    std::lock_guard<std::mutex> lock(task_lock_);
    if (stop_source_.stop_requested()) {
      throw worker_stopped("worker is stopped");
    }
    tasks_.emplace(task, std::move(result));
  }
  cv_.notify_all();
}

template <typename T, typename R> Worker<T, R>::~Worker() { stop(); }

template <typename T, typename R> void Worker<T, R>::stop() {
  stop_source_.request_stop();
  cv_.notify_all();
  if (is_worker_thread()) {
    return; // joining ourselves would deadlock; the owner joins us later
  }
  for (auto &t : worker_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

} // namespace internal
} // namespace zinferlm
