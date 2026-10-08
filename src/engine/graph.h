#include <cstdint>
#include <functional>
#include <ggml-cpp.h>
#include <memory>
#include <string>
#include <vector>

#include "layers/layers.h"

class Graph {
private:
  ggml_context *ctx_;
  std::vector<Layer *> layers_;
  ggml_tensor *input_;
  ggml_tensor *output_;
  ggml_backend_ptr backend_;
  ggml_backend_ptr cpu_backend_;
  ggml_backend_sched_ptr sched_;
  ggml_backend_buffer_ptr tensor_buffer_;
  std::vector<ggml_tensor *> intermediate_tensors_;

  // Debug logging support
  bool debug_mode_ = false;
  Graph();

public:
  static Graph &get_instance();

  ggml_context* init_context(uint64_t op_estimate);
  void free_context();
  void set_layers(std::vector<Layer *> layers);
  ggml_cgraph *build(uint32_t n_tokens);
  ggml_tensor *execute(std::vector<int32_t> input);
  ggml_backend *get_backend();

  void set_debug_mode(bool enabled) { debug_mode_ = enabled; }
};
