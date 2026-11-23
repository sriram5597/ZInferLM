#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

#include <ggml.h>

#include "engine/engine.h"
#include "engine/lifecycle.h"
#include "logging/log.h"
#include "model_impl.h"
#include "model_loader/gguf/gguf.h"
#include "qwen/model.h"
#include <zinferlm/models.h>

namespace zinferlm {

std::unique_ptr<Model> Model::instance_ = nullptr;

Model::Model() = default;
Model::~Model() = default;

Model &Model::instance() {
  if (!instance_) {
    throw std::logic_error("Model::instance() called before Model::load()");
  }
  return *instance_;
}

internal::ModelImpl &Model::impl() { return *impl_; }

bool Model::load(const char *model_path) {
  if (internal::engine_is_running()) {
    LOG_ERR("Refusing to load a model while the inference engine is running\n");
    return false;
  }
  ggml_log_set(zinferlm::internal::log_default_callback, nullptr);
  ggml_log_set(zinferlm::internal::log_default_callback, nullptr);
  int raw_fd = open(model_path, O_RDONLY);
  if (raw_fd == -1) {
    LOG_ERR("Unable to read model file: %s\n", model_path);
    return false;
  }
  std::unique_ptr<int, void (*)(int *)> fd_guard(&raw_fd,
                                                 [](int *fd) { close(*fd); });
  struct stat model_stats;
  if (fstat(raw_fd, &model_stats) == -1) {
    LOG_ERR("Unable to get file descriptor\n");
    return false;
  }
  size_t file_size = model_stats.st_size;
  if (internal::is_gguf_file(&raw_fd)) {
    internal::ModelLoader *f = internal::load_gguf_file(&raw_fd, file_size);
    instance_ = std::make_unique<Model>();
    instance_->impl_ = std::make_unique<internal::QwenModel>(f);
    instance_->impl_->load_tensors();
    instance_->size_ = instance_->impl_->size();

    internal::Engine::instance().start();
    return true;
  }
  return false;
}

bool Model::unload() {
  if (internal::engine_is_running()) {
    LOG_ERR("Refusing to unload the model: the inference engine is still "
            "running. Call Chat::stop() first.\n");
    return false;
  }
  if (instance_) {
    instance_.reset();
    LOG_INF("Model unloaded\n");
  }
  return true;
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
