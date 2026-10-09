#pragma once

#include <cstdint>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>
#include <memory>
#include <string>
#include <vector>

#include "engine/debug.h"

class KVCache;

namespace zinferlm {

class Model;

class Inference {
private:
  Model &model_;
  ggml_backend_sched_ptr sched_;

  ggml_context *ctx_ = nullptr;
  std::vector<uint8_t> ctx_buffer_;
  ggml_cgraph *gf_ = nullptr;
  ggml_tensor *output_ = nullptr;

  uint32_t cached_n_tokens_ = 0;
  bool graph_ready_ = false;

  std::unique_ptr<KVCache> cache_;

  bool debug_ = false;
  DebugCallbackData debug_cb_data_;

  ggml_context *init_context(uint64_t op_estimate);
  ggml_cgraph *new_graph(uint32_t n_tokens, uint32_t max_context_len);
  void set_output(ggml_tensor *out);
  bool prepare(uint32_t n_tokens);
  ggml_tensor *compute(std::vector<int32_t> input, int past_tokens);

  std::vector<float> predict(std::vector<int32_t> tokens, int past_tokens);

public:
  explicit Inference(Model &model);
  ~Inference();

  std::string invoke(std::string input, int max_tokens);
  void set_debug(bool enabled) { debug_ = enabled; }
};

} // namespace zinferlm
