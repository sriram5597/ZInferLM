#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <iostream>
#include <vector>

#include "debug.h"
#include "graph.h"
#include "layers/layers.h"

ggml_context *Graph::init_context(uint64_t n_op_estimate) {
  uint64_t ctx_size = n_op_estimate * ggml_tensor_overhead() +
                      ggml_graph_overhead_custom(n_op_estimate, false) + 1024;
  ggml_init_params params = {
      .mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true};
  ctx_ = ggml_init(params);
  return ctx_;
}

Graph::Graph() {
  backend_ = ggml_backend_ptr{ggml_backend_cpu_init()};

  ggml_backend_t backends[] = {backend_.get()};
  sched_ = ggml_backend_sched_ptr{
      ggml_backend_sched_new(backends, nullptr, 1, 16384, false, true)};
}

void Graph::set_layers(std::vector<Layer *> layers) { layers_ = layers; }

void Graph::free_context() { ggml_free(ctx_); }

ggml_cgraph *Graph::build(uint32_t input_size) {
  ggml_cgraph *gf = ggml_new_graph(ctx_);
  input_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, input_size);
  ggml_set_input(input_);
  ggml_set_name(input_, "input");
  ggml_tensor *current = input_;

  for (size_t i = 0; i < layers_.size(); i++) {
    layers_[i]->set_graph(gf);
    current = (*layers_[i])(current);
  }
  output_ = current;
  ggml_build_forward_expand(gf, output_);
  return gf;
}

ggml_tensor *Graph::execute(std::vector<int32_t> input) {
  ggml_cgraph *gf = build(input.size());

  // Set up debug callback with layer tensors
  DebugCallbackData cb_data = {debug_mode_, {}};
  ggml_backend_sched_set_eval_callback(sched_.get(), debug_eval_callback,
                                       &cb_data);

  // Reserve scheduler memory (MUST be before setting inputs)
  if (!ggml_backend_sched_reserve(sched_.get(), gf)) {
    std::cerr << "Failed to reserve scheduler memory" << std::endl;
    return nullptr;
  }
  ggml_backend_sched_alloc_graph(sched_.get(), gf);

  // Set input tensor
  if (debug_mode_) {
    std::cout << "Input Tokens: [ ";
    for (auto t : input)
      std::cout << t << ", ";
    std::cout << " ]\n";
  }

  ggml_backend_tensor_set(input_, input.data(), 0,
                          input.size() * sizeof(int32_t));

  // Execute with scheduler
  auto status = ggml_backend_sched_graph_compute(sched_.get(), gf);
  if (status != GGML_STATUS_SUCCESS) {
    std::cerr << "Graph compute failed with status: " << status << std::endl;
    return nullptr;
  }

  // Read final output
  size_t output_elements = ggml_nelements(output_);
  size_t output_to_print = std::min((size_t)20, output_elements);
  std::vector<float> output_data(output_to_print);
  ggml_backend_tensor_get(output_, output_data.data(), 0,
                          output_to_print * sizeof(float));

  // Find max logit
  std::vector<float> all_logits(output_elements);
  ggml_backend_tensor_get(output_, all_logits.data(), 0,
                          output_elements * sizeof(float));
  float max_val = all_logits[0];
  int max_idx = 0;
  for (int i = 1; i < all_logits.size(); i++) {
    if (all_logits[i] > max_val) {
      max_val = all_logits[i];
      max_idx = i;
    }
  }

  return output_;
}

ggml_backend *Graph::get_backend() { return backend_.get(); }

Graph &Graph::get_instance() {
  static Graph graph;
  return graph;
}
