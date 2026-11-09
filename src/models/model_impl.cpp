#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#if defined(CUDA_ENABLED) && CUDA_ENABLED
#include <ggml-cuda.h>
#endif

#include "engine/tensors.h"
#include "model_impl.h"

namespace zinferlm {
namespace internal {

ModelImpl::ModelImpl(ModelLoader *loader)
    : loader_(std::unique_ptr<ModelLoader>{loader}) {
  ggml_init_params params = {.mem_size = loader_->get_tensor_count() *
                                         ggml_tensor_overhead(),
                             .mem_buffer = nullptr,
                             .no_alloc = true};
  tensor_ctx_ = ggml_context_ptr{ggml_init(params)};

  ggml_backend_ptr cpu_backend = ggml_backend_ptr{ggml_backend_cpu_init()};
  backends_.push_back(std::move(cpu_backend));
#if defined(CUDA_ENABLED) && CUDA_ENABLED
  if (get_backend_type() == GGML_BACKEND_DEVICE_TYPE_GPU) {
    std::cout << "Using cuda backend.." << std::endl;
    ggml_backend_ptr cuda_backend = ggml_backend_ptr{ggml_backend_cuda_init(0)};
    backends_.insert(backends_.begin(), std::move(cuda_backend));
  }
#endif
}

ModelImpl::~ModelImpl() = default;

enum ggml_backend_dev_type ModelImpl::get_backend_type() {
  enum ggml_backend_dev_type backend_type = GGML_BACKEND_DEVICE_TYPE_CPU;
#if defined(CUDA_ENABLED) && CUDA_ENABLED
  backend_type = GGML_BACKEND_DEVICE_TYPE_GPU;
#endif
  return backend_type;
}

ggml_backend *ModelImpl::get_backend() { return backends_[0].get(); }

std::vector<ggml_backend *> ModelImpl::backends() const {
  std::vector<ggml_backend *> ptrs;
  ptrs.reserve(backends_.size());
  for (const auto &b : backends_) {
    ptrs.push_back(b.get());
  }
  return ptrs;
}

void ModelImpl::load_tensors() {
  for (auto &t_info : loader_->tensor_info()) {
    ggml_tensor *t = create_tensor(tensor_ctx_.get(), t_info);
    tensor_map_[t_info.name] = t;
  }
  if (get_backend_type() == GGML_BACKEND_DEVICE_TYPE_GPU) {
    std::cout << "setting buffer..." << std::endl;
    tensor_buffer_ =
        ggml_backend_buffer_ptr{ggml_backend_alloc_ctx_tensors_from_buft(
            tensor_ctx_.get(),
            ggml_backend_get_default_buffer_type(get_backend()))};
  }

  for (auto &t_info : loader_->tensor_info()) {
    ggml_tensor *t = get_tensor_(t_info.name);
    if (get_backend_type() != GGML_BACKEND_DEVICE_TYPE_CPU) {
      GGML_ASSERT(tensor_buffer_.get() != NULL && "tensor buffer not set");
      ggml_backend_tensor_set(t, loader_->get_tensor_ptr(t_info.data_offset), 0,
                              ggml_nbytes(t));
    } else {
      t->data = loader_->get_tensor_ptr(t_info.data_offset);
    }
  }
}

ggml_tensor *ModelImpl::get_tensor_(std::string name) {
  auto it = tensor_map_.find(name);
  if (it != tensor_map_.end()) {
    return it->second;
  }
  return nullptr;
}

void ModelImpl::build_mask(std::vector<ggml_fp16_t> &mask, int past_tokens,
                           const int size) {
  int filled = past_tokens + size;
  for (int i = 0; i < size; i++) {
    int pos = past_tokens + i;
    for (int j = 0; j < filled; j++) {
      float m = j <= pos ? 0.0f : -INFINITY;
      mask[i * filled + j] = ggml_fp32_to_fp16(m);
    }
  }
}

void ModelImpl::create_input_tensors(ggml_context *ctx, uint32_t n_tokens,
                                     uint32_t past_tokens) {
  ggml_tensor *input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  ggml_set_input(input);
  ggml_set_name(input, "input");

  ggml_tensor *positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  ggml_set_input(positions);
  ggml_set_name(positions, "positions");

  ggml_tensor *mask =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F16, past_tokens + n_tokens, n_tokens);
  ggml_set_input(mask);
  ggml_set_name(mask, "mask");

  ggml_tensor *lookup_rows =
      ggml_new_tensor_1d(ctx, GGML_TYPE_I32, past_tokens + n_tokens);
  ggml_set_input(lookup_rows);
  ggml_set_name(lookup_rows, "lookup_rows");

  ggml_tensor *write_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
  ggml_set_input(write_rows);
  ggml_set_name(write_rows, "write_rows");

  input_tensors_ = {input, positions, mask, lookup_rows, write_rows};
}

std::vector<ggml_tensor *> ModelImpl::get_input_tensors() {
  return input_tensors_;
}

void ModelImpl::set_inputs(const std::vector<int32_t> &tokens, int past_tokens,
                           std::vector<int32_t> cache_lookup,
                           std::vector<int64_t> write_pos) {
  ggml_tensor *input = input_tensors_[0];
  ggml_tensor *positions = input_tensors_[1];
  ggml_tensor *mask = input_tensors_[2];
  ggml_tensor *lookup_rows = input_tensors_[3];
  ggml_tensor *write_rows = input_tensors_[4];

  ggml_backend_tensor_set(input, tokens.data(), 0,
                          tokens.size() * sizeof(int32_t));

  std::vector<int32_t> pos(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) {
    pos[i] = past_tokens + (int32_t)i;
  }
  ggml_backend_tensor_set(positions, pos.data(), 0,
                          pos.size() * sizeof(int32_t));

  const int filled = past_tokens + tokens.size();
  std::vector<ggml_fp16_t> mask_data((size_t)filled * tokens.size());
  build_mask(mask_data, past_tokens, tokens.size());
  ggml_backend_tensor_set(mask, mask_data.data(), 0,
                          mask_data.size() * sizeof(ggml_fp16_t));

  ggml_backend_tensor_set(lookup_rows, cache_lookup.data(), 0,
                          cache_lookup.size() * sizeof(int32_t));
  ggml_backend_tensor_set(write_rows, write_pos.data(), 0,
                          write_pos.size() * sizeof(int64_t));
}

zinferlm::model_info_t ModelImpl::info() const { return loader_->info(); }

zinferlm::tokenizer_info_t ModelImpl::tokenizer_info() const {
  return loader_->tokenizer_info();
}

std::vector<zinferlm::tensor_info_t> ModelImpl::tensor_info() const {
  return loader_->tensor_info();
}

zinferlm::model_config_t ModelImpl::config() const {
  return loader_->model_config();
}

} // namespace internal
} // namespace zinferlm
