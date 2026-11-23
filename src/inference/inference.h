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
  ModelImpl &model_impl_;

  int max_batches_ = 16;
  uint32_t prefill_chunksize_ = 16;
  bool reserved = false;
  bool measure_only_ = false;
  size_t size_ = 0;
  size_t metadata_size_ = 0;
  std::vector<size_t> backend_sizes_;

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
  bool prepare(ggml_tensor* additional_input);
  void compute(const model_input_t &input);
  bool reserve_memory();

public:
  explicit Inference(KVCache *cache, zinferlm::Model &model,
                     int max_batches = 16, uint32_t chunk_size = 16,
                     bool measure_only = false);
  explicit Inference(KVCache *cache, ModelImpl &model_impl,
                     int max_batches = 16, uint32_t chunk_size = 16,
                     bool measure_only = false);
  ~Inference();
  std::string invoke(std::string input, uint32_t max_tokens);
  std::pair<std::vector<float>, int64_t>
  predict(const model_input_t &input, size_t n_tokens, size_t total_filled);

  // Bytes reserved for compute buffers (all backends).
  size_t size() const { return size_; }
  // Host bytes used by the measure graph.
  size_t metadata_size() const { return metadata_size_; }
  // Per-backend compute-buffer bytes, in model backend order.
  const std::vector<size_t> &backend_sizes() const { return backend_sizes_; }

  void set_debug(bool enabled) { debug_ = enabled; }
  void set_sampler(Sampler &sampler) { sampler_ = sampler; }
};

} // namespace internal
} // namespace zinferlm
