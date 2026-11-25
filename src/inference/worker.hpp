#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <prometheus/counter.h>
#include <prometheus/detail/core_export.h>
#include <prometheus/family.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <queue>
#include <string>

#include "inference/inference.h"
#include "kv_cache/cache.h"
#include "sampling/samplers.h"
#include "worker/worker.hpp"

namespace zinferlm {
namespace internal {

struct inference_task_t {
  std::string task_id;
  sampler_params_t sample_config;
  std::string input;
  uint64_t max_tokens;
};

struct inference_worker_metrics_t {
  prometheus::Histogram* tokenizer_duration;
};

class InferenceWorker : public Worker<inference_task_t, std::string> {
private:
  friend class Engine;
  std::unique_ptr<KVCache> cache_;
  std::unique_ptr<Inference> inference_;
  std::mutex seq_lock_;
  int max_batches_;
  inference_worker_metrics_t metrics_;

  void execute_task_() override;
  InferenceWorker();
  std::vector<int32_t> tokenize(const std::string &task_id, std::string input);

  void init_metrics();
};

} // namespace internal
} // namespace zinferlm
