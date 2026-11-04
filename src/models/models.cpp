#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <unistd.h>

#include <ggml.h>

#include "model_impl.h"
#include "model_loader/gguf/gguf.h"
#include "qwen/model.h"
#include <zinferlm/models.h>

namespace {
void log_filter(enum ggml_log_level level, const char *text, void *user_data) {
  (void)user_data;
  if (level >= GGML_LOG_LEVEL_INFO) {
    fputs(text, stderr);
    fflush(stderr);
  }
}
} // namespace

namespace zinferlm {

std::unique_ptr<Model> Model::instance_ = nullptr;

Model::Model() = default;
Model::~Model() = default;

Model &Model::instance() {
  assert(instance_ && "Model not loaded. Call load() first.");
  return *instance_;
}

internal::ModelImpl &Model::impl() { return *impl_; }

bool Model::load(const char *model_path) {
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
  if (internal::is_gguf_file(&raw_fd)) {
    internal::ModelLoader *f = internal::load_gguf_file(&raw_fd, file_size);
    instance_ = std::make_unique<Model>();
    instance_->impl_ = std::make_unique<internal::QwenModel>(f);
    instance_->impl_->load_tensors();
    return true;
  }
  return false;
}

void Model::unload() {
  if (instance_) {
    instance_.reset();
    std::cout << "Model unloaded" << std::endl;
  }
}

model_info_t Model::info() const { return impl_->info(); }

tokenizer_info_t Model::tokenizer_info() const {
  return impl_->tokenizer_info();
}

std::vector<tensor_info_t> Model::tensor_info() const {
  return impl_->tensor_info();
}

model_config_t Model::config() const { return impl_->config(); }

void Model::summary() { impl_->summary(); }

} // namespace zinferlm
