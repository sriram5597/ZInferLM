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
            tensor_ctx_.get(), ggml_backend_get_default_buffer_type(
                                   get_backend()))};
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
