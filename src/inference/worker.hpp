#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <string>

#include "inference/inference.h"
#include "kv_cache/cache.h"
#include "worker/worker.hpp"

namespace zinferlm {
namespace internal {

struct inference_task_t {
  zinferlm::Model &model;
  bool debug;
  Sampler sampler;
  std::string input;
  uint64_t max_tokens;
};

class InferenceWorker : public Worker<inference_task_t, std::string> {
private:
  std::unique_ptr<KVCache> cache_;
  std::unique_ptr<Inference> inference_;
  void execute_task_(inference_task_t &task,
                     std::promise<std::string> &result) override;
  InferenceWorker();
  std::vector<int32_t> tokenize(std::string input);

public:
  static InferenceWorker &instance();
};

} // namespace internal
} // namespace zinferlm
