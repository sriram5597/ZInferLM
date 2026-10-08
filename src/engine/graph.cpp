#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <iostream>
#include <ostream>
#include <utility>
#include <vector>

#if defined(CUDA_ENABLED) && CUDA_ENABLED
#include <ggml-cuda.h>
#endif

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

Backend Graph::get_backend_type() {
  Backend backend_type = Backend::GGML_BACKEND_DEVICE_TYPE_CPU;
#if defined(CUDA_ENABLED) && CUDA_ENABLED
  backend_type = Backend::GGML_BACKEND_DEVICE_TYPE_GPU;
#endif
  return backend_type;
}

Graph::Graph() {
  ggml_backend_ptr cpu_backend = ggml_backend_ptr{ggml_backend_cpu_init()};
  backends_.push_back(std::move(cpu_backend));
#if defined(CUDA_ENABLED) && CUDA_ENABLED
  if (get_backend_type() == Backend::GGML_BACKEND_DEVICE_TYPE_GPU) {
    std::cout << "Using cuda backend.." << std::endl;
    ggml_backend_ptr cuda_backend = ggml_backend_ptr{ggml_backend_cuda_init(0)};
    backends_.insert(backends_.begin(), std::move(cuda_backend));
  }
#endif
  std::vector<ggml_backend *> backend_ptrs;
  for (auto &b : backends_) {
    backend_ptrs.push_back(b.get());
  }

  sched_ = ggml_backend_sched_ptr{ggml_backend_sched_new(
      backend_ptrs.data(), nullptr, backend_ptrs.size(), 16384, false, true)};
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

  // Pin the token-ID input tensor to the primary backend. ggml places
  // GGML_TENSOR_FLAG_INPUT tensors on the last (CPU) backend by default and
  // inserts a host->device copy, but that copy path is unreliable across our
  // repeated per-token graph rebuilds. Placing the input on the primary backend
  // (the GPU when CUDA is enabled) writes token IDs directly into the buffer the
  // embedding lookup kernel reads.
  ggml_backend_sched_set_tensor_backend(sched_.get(), input_, backends_[0].get());

  // Install the debug eval callback only when requested, and keep its state in a
  // member so the scheduler never holds a dangling stack pointer.
  if (debug_mode_) {
    debug_cb_data_ = {debug_mode_, {}};
    ggml_backend_sched_set_eval_callback(sched_.get(), debug_eval_callback,
                                         &debug_cb_data_);
  } else {
    ggml_backend_sched_set_eval_callback(sched_.get(), nullptr, nullptr);
  }

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
  return output_;
}

ggml_backend *Graph::get_backend() { return backends_[0].get(); }

Graph &Graph::get_instance() {
  static Graph graph;
  return graph;
}
