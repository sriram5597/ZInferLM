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

#include "engine/measure.h"
#include "engine/tensors.h"
#include "logging/log.h"
#include "model_impl.h"

namespace zinferlm {
namespace internal {

ModelImpl::ModelImpl(ModelLoader *loader, bool cpu_only)
    : loader_(std::unique_ptr<ModelLoader>{loader}), cpu_only_(cpu_only) {
  ggml_init_params params = {.mem_size = loader_->get_tensor_count() *
                                         ggml_tensor_overhead(),
                             .mem_buffer = nullptr,
                             .no_alloc = true};
  tensor_ctx_ = ggml_context_ptr{ggml_init(params)};

  ggml_backend_ptr cpu_backend = ggml_backend_ptr{ggml_backend_cpu_init()};
  backends_.push_back(std::move(cpu_backend));
#if defined(CUDA_ENABLED) && CUDA_ENABLED
  if (!cpu_only_ && get_backend_type() == GGML_BACKEND_DEVICE_TYPE_GPU) {
    LOG_INF("Using cuda backend..\n");
    ggml_backend_ptr cuda_backend = ggml_backend_ptr{ggml_backend_cuda_init(0)};
    if (cuda_backend) {
      backends_.insert(backends_.begin(), std::move(cuda_backend));
    } else {
      LOG_WRN("cuda backend unavailable, falling back to cpu\n");
    }
  }
#endif
}

ModelImpl::~ModelImpl() = default;

bool ModelImpl::use_gpu() const {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
  return !cpu_only_ &&
         get_backend_type() == GGML_BACKEND_DEVICE_TYPE_GPU;
#else
  return false;
#endif
}

size_t ModelImpl::tensor_metadata_size() const {
  return loader_->get_tensor_count() * ggml_tensor_overhead();
}

std::vector<ggml_tensor *> ModelImpl::tensors() const {
  std::vector<ggml_tensor *> out;
  out.reserve(tensor_map_.size());
  for (const auto &[name, tensor] : tensor_map_) {
    out.push_back(tensor);
  }
  return out;
}

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

void ModelImpl::create_tensor_structs() {
  model_size_ = 0;
  for (auto &t_info : loader_->tensor_info()) {
    ggml_tensor *t = create_tensor(tensor_ctx_.get(), t_info);
    tensor_map_[t_info.name] = t;
    model_size_ += ggml_nbytes(t);
  }
}

void ModelImpl::mark_tensors_external() {
  for (auto &[name, tensor] : tensor_map_) {
    mark_external_tensor(tensor);
  }
}

void ModelImpl::load_tensors() {
  if (tensor_map_.empty()) {
    create_tensor_structs();
  }
  if (use_gpu()) {
    LOG_DBG("setting buffer...\n");
    tensor_buffer_ =
        ggml_backend_buffer_ptr{ggml_backend_alloc_ctx_tensors_from_buft(
            tensor_ctx_.get(),
            ggml_backend_get_default_buffer_type(get_backend()))};
  }

  for (auto &t_info : loader_->tensor_info()) {
    ggml_tensor *t = get_tensor_(t_info.name);
    if (use_gpu()) {
      GGML_ASSERT(tensor_buffer_.get() != NULL && "tensor buffer not set");
      ggml_backend_tensor_set(t, loader_->get_tensor_ptr(t_info.data_offset), 0,
                              ggml_nbytes(t));
    } else {
      t->data = loader_->get_tensor_ptr(t_info.data_offset);
    }
  }
  LOG_INF("Tensors loaded: tensors=%zu model_size=%zu\n", tensor_map_.size(),
          (size_t)model_size_);
}

ggml_tensor *ModelImpl::get_tensor_(std::string name) {
  auto it = tensor_map_.find(name);
  if (it != tensor_map_.end()) {
    return it->second;
  }
  return nullptr;
}

void ModelImpl::create_input_tensors(ggml_context *ctx, size_t n_tokens,
                                     size_t total_filled) {
  ggml_tensor *input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  ggml_set_input(input);
  ggml_set_name(input, "input");

  ggml_tensor *positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  ggml_set_input(positions);
  ggml_set_name(positions, "positions");

  ggml_tensor *mask =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F16, total_filled, n_tokens);
  ggml_set_input(mask);
  ggml_set_name(mask, "mask");

  ggml_tensor *lookup_rows =
      ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_filled);
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

void ModelImpl::set_inputs(const model_input_t &input) {
  ggml_tensor *tokens = input_tensors_[0];
  ggml_tensor *positions = input_tensors_[1];
  ggml_tensor *mask = input_tensors_[2];
  ggml_tensor *lookup_rows = input_tensors_[3];
  ggml_tensor *write_rows = input_tensors_[4];

  ggml_backend_tensor_set(tokens, input.tokens.data(), 0,
                          input.tokens.size() * sizeof(int32_t));
  ggml_backend_tensor_set(positions, input.positions.data(), 0,
                          input.positions.size() * sizeof(int32_t));
  ggml_backend_tensor_set(mask, input.causal_mask.data(), 0,
                          input.causal_mask.size() * sizeof(ggml_fp16_t));
  ggml_backend_tensor_set(lookup_rows, input.cache_lookup_rows.data(), 0,
                          input.cache_lookup_rows.size() * sizeof(int32_t));
  ggml_backend_tensor_set(write_rows, input.cache_write_rows.data(), 0,
                          input.cache_write_rows.size() * sizeof(int64_t));
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
