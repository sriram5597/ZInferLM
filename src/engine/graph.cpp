#include <cstdint>
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

// Debug callback data structure
struct DebugCallbackData {
    bool debug_mode;
    std::vector<uint8_t> buffer;
    std::vector<ggml_tensor*> layer_tensors;
};

// Helper function to print tensor values
static void print_tensor_values(const uint8_t *data, ggml_type type, const int64_t *ne, const size_t *nb, int max_elements) {
    int count = 0;
    for (int i3 = 0; i3 < ne[3] && count < max_elements; i3++) {
        for (int i2 = 0; i2 < ne[2] && count < max_elements; i2++) {
            for (int i1 = 0; i1 < ne[1] && count < max_elements; i1++) {
                for (int i0 = 0; i0 < ne[0] && count < max_elements; i0++) {
                    size_t offset = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float value = 0.0f;
                    
                    if (type == GGML_TYPE_F32) {
                        value = *reinterpret_cast<const float*>(data + offset);
                    } else if (type == GGML_TYPE_F16) {
                        uint16_t f16_val = *reinterpret_cast<const uint16_t*>(data + offset);
                        value = ggml_fp16_to_fp32(f16_val);
                    } else if (type == GGML_TYPE_I32) {
                        value = static_cast<float>(*reinterpret_cast<const int32_t*>(data + offset));
                    } else {
                        std::cout << "[unsupported type] ";
                        return;
                    }
                    
                    std::cout << std::fixed << std::setprecision(4) << value << " ";
                    count++;
                }
            }
        }
    }
}

// Eval callback - called for each tensor after computation
static bool debug_eval_callback(ggml_tensor *tensor, bool ask, void *user_data) {
    auto *cb_data = static_cast<DebugCallbackData*>(user_data);
    
    // Check if this tensor is one of our layer outputs
    bool is_layer_output = false;
    for (auto* layer_tensor : cb_data->layer_tensors) {
        if (tensor == layer_tensor) {
            is_layer_output = true;
            break;
        }
    }
    
    if (ask) {
        // Only request layer outputs
        return cb_data->debug_mode && is_layer_output;
    }
    
    // ask=false: tensor has been computed, read its data
    if (!cb_data->debug_mode || !is_layer_output) {
        return true;
    }
    
    // Skip quantized types (can't read floats directly)
    if (ggml_is_quantized(tensor->type)) {
        return true;
    }
    
    // Print tensor info
    const char *name = ggml_get_name(tensor);
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
    std::cout << "    values: ";
    print_tensor_values(data_ptr, tensor->type, tensor->ne, tensor->nb, 10);
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

  intermediate_tensors_.clear();
  for (size_t i = 0; i < layers_.size(); i++) {
    current = (*layers_[i])(current);
    intermediate_tensors_.push_back(current);
    if (!layers_[i]->name.empty()) {
      ggml_set_name(current, layers_[i]->name.c_str());
    }
  }
  output_ = current;
  ggml_set_name(output_, "output");
  ggml_build_forward_expand(gf, output_);
  return gf;
}

ggml_tensor *Graph::execute(std::vector<uint32_t> input) {
  ggml_cgraph *gf = build(input.size());
  
  // Set up debug callback with layer tensors
  DebugCallbackData cb_data = {debug_mode_, {}, intermediate_tensors_};
  ggml_backend_sched_set_eval_callback(sched_.get(), debug_eval_callback, &cb_data);
  
  // Reserve scheduler memory (MUST be before setting inputs)
  if (!ggml_backend_sched_reserve(sched_.get(), gf)) {
    std::cerr << "Failed to reserve scheduler memory" << std::endl;
    return nullptr;
  }
  ggml_backend_sched_alloc_graph(sched_.get(), gf);
  
  // Set input tensor
  ggml_backend_tensor_set(input_, input.data(), 0,
                          input.size() * sizeof(uint32_t));

  std::cout << "Input tokens: ";
  for (size_t j = 0; j < input.size() && j < 10; j++) {
    std::cout << input[j] << " ";
  }
  std::cout << std::endl;

  // Execute with scheduler
  auto status = ggml_backend_sched_graph_compute(sched_.get(), gf);
  if (status != GGML_STATUS_SUCCESS) {
    std::cerr << "Graph compute failed with status: " << status << std::endl;
    return nullptr;
  }

  // Read final output
  std::cout << "\n[OUTPUT] Final logits (first 20): ";
  size_t output_elements = ggml_nelements(output_);
  size_t output_to_print = std::min((size_t)20, output_elements);
  std::vector<float> output_data(output_to_print);
  ggml_backend_tensor_get(output_, output_data.data(), 0, output_to_print * sizeof(float));
  for (size_t j = 0; j < output_to_print; j++) {
    std::cout << output_data[j] << " ";
  }
  std::cout << std::endl;

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
  std::cout << "[OUTPUT] Max logit at index " << max_idx << ": " << max_val << std::endl;

  return output_;
}
