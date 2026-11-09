#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <zinferlm/models.h>

#include "model_loader/loader.h"

namespace zinferlm {
namespace internal {

class KVCache;

// Internal model implementation interface. Concrete models (e.g. QwenModel)
// derive from this and implement the graph-building surface. This is the
// engine-facing counterpart of the public Model facade and is never exposed.
class ModelImpl {
public:
  explicit ModelImpl(ModelLoader *loader);
  virtual ~ModelImpl();

  // Graph-building surface (implemented by concrete models).
  virtual ggml_tensor *build_graph(ggml_context *ctx, ggml_cgraph *gf,
                                   KVCache *cache) = 0;
  virtual void summary() = 0;

  void create_input_tensors(ggml_context *ctx, uint32_t n_tokens,
                            uint32_t past_tokens);
  std::vector<ggml_tensor *> get_input_tensors();
  void set_inputs(const std::vector<int32_t> &tokens, int past_tokens,
                  std::vector<int32_t> cache_lookup,
                  std::vector<int64_t> write_rows);

  // Metadata passthrough.
  zinferlm::model_info_t info() const;
  zinferlm::tokenizer_info_t tokenizer_info() const;
  std::vector<zinferlm::tensor_info_t> tensor_info() const;
  zinferlm::model_config_t config() const;

  // Backend access.
  ggml_backend *get_backend();
  std::vector<ggml_backend *> backends() const;
  static enum ggml_backend_dev_type get_backend_type();

  // Tensor loading.
  void load_tensors();

protected:
  ggml_tensor *get_tensor_(std::string name);

  ggml_context_ptr tensor_ctx_;
  std::unordered_map<std::string, ggml_tensor *> tensor_map_;
  std::unique_ptr<ModelLoader> loader_;
  std::vector<ggml_backend_ptr> backends_;
  std::vector<ggml_tensor *> input_tensors_;
  void build_mask(std::vector<ggml_fp16_t> &mask, const int filled,
                  const int size);

private:
  ggml_backend_buffer_ptr tensor_buffer_;
};

} // namespace internal
} // namespace zinferlm
