#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <iostream>
#include <memory>
#include <ostream>
#include <sys/stat.h>
#include <unistd.h>

#if defined(CUDA_ENABLED) && CUDA_ENABLED
#include <ggml-cuda.h>
#endif

#include "engine/tensors.h"
#include "ggml-cpp.h"
#include "model_loader/gguf/gguf.h"
#include "qwen/model.h"
#include <zinferlm/models.h>

static_assert(sizeof(struct ggml_tensor) > 0, "ggml integration check");

namespace {
void log_filter(enum ggml_log_level level, const char *text, void *user_data) {
  (void)user_data;
  if (level >= GGML_LOG_LEVEL_INFO) {
    fputs(text, stderr);
    fflush(stderr);
  }
}
} // namespace

std::unique_ptr<zinferlm::Model> zinferlm::Model::instance_ = nullptr;

zinferlm::Model &zinferlm::Model::instance() {
  assert(instance_ && "Model not loaded. Call load() first.");
  return *instance_;
}

zinferlm::Model::Model(zinferlm::ModelLoader *loader)
    : loader_(std::unique_ptr<zinferlm::ModelLoader>{loader}) {
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

Backend zinferlm::Model::get_backend_type() {
  Backend backend_type = GGML_BACKEND_DEVICE_TYPE_CPU;
#if defined(CUDA_ENABLED) && CUDA_ENABLED
  backend_type = GGML_BACKEND_DEVICE_TYPE_GPU;
#endif
  return backend_type;
}

ggml_backend *zinferlm::Model::get_backend() { return backends_[0].get(); }

std::vector<ggml_backend *> zinferlm::Model::backends() const {
  std::vector<ggml_backend *> ptrs;
  ptrs.reserve(backends_.size());
  for (const auto &b : backends_) {
    ptrs.push_back(b.get());
  }
  return ptrs;
}

bool zinferlm::Model::load(const char *model_path) {
  ggml_log_set(log_filter, nullptr);
  int raw_fd = open(model_path, O_RDONLY);
  if (raw_fd == -1) {
    std::cerr << "Unable to read model file: " << model_path << std::endl;
    return false;
  }
  std::unique_ptr<int, void (*)(int *)> fd_guard(&raw_fd,
                                                 [](int *fd) { close(*fd); });
  struct stat model_stats;
  if (fstat(raw_fd, &model_stats) == -1) {
    std::cerr << "Unable to get file descriptor" << std::endl;
    return false;
  }
  size_t file_size = model_stats.st_size;
  if (is_gguf_file(&raw_fd)) {
    ModelLoader *f = load_gguf_file(&raw_fd, file_size);
    instance_ = std::make_unique<QwenModel>(f);
    instance_->load_tensors_();
    return true;
  }
  return false;
}

void zinferlm::Model::load_tensors_() {
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

ggml_tensor *zinferlm::Model::get_tensor_(std::string name) {
  auto it = tensor_map_.find(name);
  if (it != tensor_map_.end()) {
    return it->second;
  }
  return nullptr;
}

zinferlm::model_info_t zinferlm::Model::info() const { return loader_->info(); }

zinferlm::tokenizer_info_t zinferlm::Model::tokenizer_info() const {
  return loader_->tokenizer_info();
}

std::vector<zinferlm::tensor_info_t> zinferlm::Model::tensor_info() const {
  return loader_->tensor_info();
}

zinferlm::model_config_t zinferlm::Model::config() const {
  return loader_->model_config();
}
