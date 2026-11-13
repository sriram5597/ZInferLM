#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>
#include <zinferlm/models.h>

#include "engine/debug.h"
#include "kv_cache/cache.h"
#include "models/model_impl.h"
#include "sampling/samplers.h"

namespace zinferlm {
namespace internal {

class Inference {
private:
  ggml_backend_sched_ptr sched_;
  Sampler sampler_;
  Model &model_;
  ModelImpl &model_impl_;

  uint32_t prefill_chunksize_ = 512;
  bool reserved = false;

  ggml_context *ctx_ = nullptr;
  std::vector<uint8_t> ctx_buffer_;
  ggml_cgraph *gf_ = nullptr;
  ggml_tensor *output_ = nullptr;

  KVCache *cache_;

  bool debug_ = false;
  DebugCallbackData debug_cb_data_;

  ggml_context *init_context(uint64_t op_estimate);
  ggml_cgraph *new_graph(size_t n_tokens, size_t total_filled);
  void set_output(ggml_tensor *out);
  bool prepare();
  void compute(const model_input_t &input);
  bool reserve_memory();

public:
  explicit Inference(KVCache *cache, zinferlm::Model &model);
  ~Inference();
  std::string invoke(std::string input, uint32_t max_tokens);
  std::pair<std::vector<float>, int64_t>
  predict(const model_input_t &input, size_t n_tokens, size_t total_filled);

  void set_debug(bool enabled) { debug_ = enabled; }
  void set_sampler(Sampler &sampler) { sampler_ = sampler; }
};

} // namespace internal
} // namespace zinferlm
