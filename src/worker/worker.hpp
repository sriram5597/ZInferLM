#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

// Thrown when a task is submitted to a stopped worker, or when a queued task is
// cancelled during shutdown. Delivered either synchronously from add_task(), or
// asynchronously through the task's std::future.
class worker_stopped : public std::runtime_error {
public:
  explicit worker_stopped(const std::string &msg) : std::runtime_error(msg) {}
};

template <typename T, typename R> class Worker {
protected:
  std::thread worker_thread_;
  std::atomic<bool> running_{false};
  std::condition_variable cv_;
  std::mutex task_lock_;
  std::queue<std::pair<T, std::promise<R>>> tasks_;
  virtual void execute_task_(T &task, std::promise<R> &result) = 0;
  Worker();

public:
  ~Worker();
  void run();
  void add_task(T &task, std::promise<R> result);
  void stop();
};

template <typename T, typename R> Worker<T, R>::Worker() {
  running_.store(true);
  worker_thread_ = std::thread([this]() { this->run(); });
}

template <typename T, typename R> void Worker<T, R>::run() {
  while (true) {
    std::unique_lock<std::mutex> lock(task_lock_);
    cv_.wait(lock, [this] { return !tasks_.empty() || !running_; });

    if (!running_) {
      // Shutdown: cancel every queued task so no caller's future.get() hangs.
      while (!tasks_.empty()) {
        tasks_.front().second.set_exception(
            std::make_exception_ptr(worker_stopped("worker stopped")));
        tasks_.pop();
      }
      break;
    }

    std::pair<T, std::promise<R>> task = std::move(tasks_.front());
    tasks_.pop();
    lock.unlock();

    execute_task_(task.first, task.second);
  }
}

template <typename T, typename R>
void Worker<T, R>::add_task(T &task, std::promise<R> result) {
  {
    std::lock_guard<std::mutex> lock(task_lock_);
    if (!running_.load()) {
      throw worker_stopped("worker is stopped");
    }
    tasks_.emplace(task, std::move(result));
  }
  cv_.notify_one();
}

template <typename T, typename R> Worker<T, R>::~Worker() { stop(); }

template <typename T, typename R> void Worker<T, R>::stop() {
  {
    std::lock_guard<std::mutex> lock(task_lock_);
    if (!running_.load()) {
      return;
    }
    running_.store(false);
  }
  cv_.notify_all();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}
