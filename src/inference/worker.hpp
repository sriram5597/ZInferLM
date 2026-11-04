#pragma once

#include <cstdint>
#include <future>
#include <string>

#include "inference/inference.h"
#include "worker/worker.hpp"

namespace zinferlm {
namespace internal {

struct inference_task_t {
  std::string run_id;
  zinferlm::Model &model;
  bool debug;
  Sampler sampler;
  std::string input;
  uint64_t max_tokens;
};

class InferenceWorker : public Worker<inference_task_t, std::string> {
private:
  void execute_task_(inference_task_t &task,
                     std::promise<std::string> &result) override;
  InferenceWorker() = default;

public:
  static InferenceWorker &instance();
};

} // namespace internal
} // namespace zinferlm
