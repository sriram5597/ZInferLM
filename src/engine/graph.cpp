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
#include <iomanip>

#include "graph.h"
#include "layers/layers.h"

#define INDENT "    "

// Debug callback data structure
struct DebugCallbackData {
    bool debug_mode;
    std::vector<uint8_t> buffer;
};


static float common_ggml_get_float_value(const uint8_t * data,
                           ggml_type       type,
                           const size_t *  nb,
                           size_t          i0,
                           size_t          i1,
                           size_t          i2,
                           size_t          i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    float  v;
    if (type == GGML_TYPE_F16) {
        v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
    } else if (type == GGML_TYPE_F32) {
        v = *(const float *) &data[i];
    } else if (type == GGML_TYPE_I64) {
        v = (float) *(const int64_t *) &data[i];
    } else if (type == GGML_TYPE_I32) {
        v = (float) *(const int32_t *) &data[i];
    } else if (type == GGML_TYPE_I16) {
        v = (float) *(const int16_t *) &data[i];
    } else if (type == GGML_TYPE_I8) {
        v = (float) *(const int8_t *) &data[i];
    } else if (type == GGML_TYPE_BF16) {
        v = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
    } else {
        GGML_ABORT("fatal error");
    }
    return v;
}

// Helper function to print tensor values
static void print_tensor_values(const uint8_t *data, ggml_type type, const int64_t *ne, const size_t *nb, int n) {
GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum += v;
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        std::printf(INDENT "[\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) {
                std::printf(INDENT INDENT "..., \n");
                i2 = ne[2] - n;
            }
            std::printf(INDENT INDENT "[\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) {
                    std::printf(INDENT INDENT INDENT "..., \n");
                    i1 = ne[1] - n;
                }
                std::printf(INDENT INDENT INDENT "[");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) {
                        std::printf("   ..., ");
                        i0 = ne[0] - n;
                    }
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    std::printf("%12.4f", v);
                    if (i0 < ne[0] - 1) {
                        std::printf(", ");
                    }
                }
                std::printf("  ],\n");
            }
            std::printf(INDENT INDENT "],\n");
        }
        std::printf(INDENT "]\n");
        std::printf(INDENT "sum = %f\n", sum);
    }


    }

// Eval callback - called for each tensor after computation
static bool debug_eval_callback(ggml_tensor *tensor, bool ask, void *user_data) {
    auto *cb_data = static_cast<DebugCallbackData*>(user_data);
    bool is_layer_output = true;
    const char *name = ggml_get_name(tensor);
   
    if (ask) {
        return cb_data->debug_mode && (is_layer_output);
    }
    
    if (!cb_data->debug_mode || (!is_layer_output)) {
        return true;
    }
    
    if (ggml_is_quantized(tensor->type)) {
        return true;
    }
    
    const char *op = ggml_op_desc(tensor);
    
    std::cout << "[DEBUG] " << name << " = " << op 
              << " | type: " << ggml_type_name(tensor->type)
              << " | shape: [" << tensor->ne[0];
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] > 1) {
            std::cout << ", " << tensor->ne[i];
        }
    }
    std::cout << "]" << std::endl;
    
    // Read tensor data
    size_t nbytes = ggml_nbytes(tensor);
    cb_data->buffer.resize(nbytes);
    
    // Check if tensor is on host
    bool is_host = ggml_backend_buffer_is_host(tensor->buffer);
    uint8_t *data_ptr;
    
    if (is_host) {
        data_ptr = static_cast<uint8_t*>(tensor->data);
    } else {
        // Copy from device to host
        ggml_backend_tensor_get(tensor, cb_data->buffer.data(), 0, nbytes);
        data_ptr = cb_data->buffer.data();
    }
    
    // Print first 10 values
    std::cout << "values:\n";
    print_tensor_values(data_ptr, tensor->type, tensor->ne, tensor->nb, 3);
    std::cout << std::endl;
    
    return true;
}

ggml_context_ptr init_engine(uint64_t n_op_estimate) {
  uint64_t ctx_size = n_op_estimate * ggml_tensor_overhead() +
                      ggml_graph_overhead_custom(n_op_estimate, false) + 1024;
  ggml_init_params params = {
      .mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true};
  return ggml_context_ptr{ggml_init(params)};
}

Graph::Graph(ggml_context *c, std::vector<Layer *> l) : ctx_(c), layers_(l) {
  backend_ = ggml_backend_ptr{ggml_backend_cpu_init()};
  
  // Create scheduler with the backend
  ggml_backend_t backends[] = {backend_.get()};
  sched_ = ggml_backend_sched_ptr{
      ggml_backend_sched_new(backends, nullptr, 1, 16384, false, true)};
};

ggml_cgraph *Graph::build(uint32_t input_size) {
  ggml_cgraph *gf = ggml_new_graph(ctx_);
  input_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, input_size);
  ggml_set_input(input_);
  ggml_set_name(input_, "input");
  ggml_tensor *current = input_;

  for (size_t i = 0; i < layers_.size(); i++) {
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
  ggml_backend_sched_set_eval_callback(sched_.get(), debug_eval_callback, &cb_data);
  
  // Reserve scheduler memory (MUST be before setting inputs)
  if (!ggml_backend_sched_reserve(sched_.get(), gf)) {
    std::cerr << "Failed to reserve scheduler memory" << std::endl;
    return nullptr;
  }
  ggml_backend_sched_alloc_graph(sched_.get(), gf);
  
  // Set input tensor
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
  ggml_backend_tensor_get(output_, output_data.data(), 0, output_to_print * sizeof(float));

  // Find max logit
  std::vector<float> all_logits(output_elements);
  ggml_backend_tensor_get(output_, all_logits.data(), 0, output_elements * sizeof(float));
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
