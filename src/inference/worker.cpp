#include <cinttypes>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engine/engine.h"
#include "logging/log.h"
#include "scheduler.hpp"
#include "worker.hpp"
#include "worker/worker.hpp"
#include "zinferlm/models.h"
#include "zinferlm/tokenizer.h"

namespace zinferlm {
namespace internal {

InferenceWorker::InferenceWorker()
    : zinferlm::internal::Worker<inference_task_t, std::string>(4) {}

std::vector<int32_t> InferenceWorker::tokenize(const std::string &task_id,
                                               std::string input) {
  Tokenizer tokenizer = Tokenizer::for_model(zinferlm::Model::instance());
  LOG_INF("tokenizer started: task_id=%s input_chars=%zu\n", task_id.c_str(),
          input.size());
  std::vector<int32_t> tokens = tokenizer.tokenize(input);
  LOG_INF("tokenizer completed: task_id=%s tokens=%zu\n", task_id.c_str(),
          tokens.size());
  if (tokens.empty()) {
    LOG_WRN("tokenizer produced 0 tokens for task_id=%s\n", task_id.c_str());
  }
  if (log_get_verbosity_thold() >= LOG_LEVEL_DEBUG) {
    std::string tokens_str;
    for (auto t : tokens) {
      tokens_str += std::to_string(t) + ", ";
    }
    LOG_DBG("prompt tokens: task_id=%s [ %s]\n", task_id.c_str(),
            tokens_str.c_str());
  }
  return tokens;
}

void InferenceWorker::execute_task_() {
  std::pair<inference_task_t, std::promise<std::string>> task;
  {
    std::lock_guard<std::mutex> lock(task_lock_);
    if (tasks_.empty()) {
      return;
    }
    task = std::move(tasks_.front());
    tasks_.pop();
  }
  LOG_INF("request received: task_id=%s input_chars=%zu max_tokens=%" PRIu64
          "\n",
          task.first.task_id.c_str(), task.first.input.size(),
          task.first.max_tokens);

  std::vector<int32_t> tokens;
  try {
    tokens = tokenize(task.first.task_id, task.first.input);
  } catch (...) {
    // Keep the failure inside the task's future: an exception escaping a
    // std::jthread would terminate the process.
    LOG_ERR("tokenization failed: task_id=%s\n", task.first.task_id.c_str());
    task.second.set_exception(std::current_exception());
    return;
  }

  seq_info_t seq_info = {.seq_id = task.first.task_id,
                         .past_tokens = 0,
                         .seq_len = 0,
                         .max_tokens = task.first.max_tokens,
                         .sample_config = task.first.sample_config,
                         .promise = std::move(task.second),
                         .inp_tokens = tokens};

  // The promise now belongs to seq_info: if submission throws, seq_info's
  // destructor reports a broken promise to the caller instead of crashing.
  try {
    std::shared_ptr<InferenceScheduler> scheduler =
        Engine::instance().scheduler();
    scheduler->submit(std::move(seq_info));
  } catch (const std::exception &e) {
    LOG_ERR("task submission failed: task_id=%s error=%s\n",
            task.first.task_id.c_str(), e.what());
    return;
  }
  LOG_INF("task %s submitted to scheduler\n", task.first.task_id.c_str());
}
} // namespace internal
} // namespace zinferlm
