#pragma once

#include <cstdint>
#include <memory>
#include <prometheus/family.h>
#include <prometheus/gauge.h>
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

struct model_metrics_t {
  prometheus::Family<prometheus::Gauge> *model_size;
};

struct model_input_t {
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  std::vector<ggml_fp16_t> causal_mask;
  std::vector<int32_t> cache_lookup_rows;
  std::vector<int64_t> cache_write_rows;
  std::vector<int32_t> offsets;
};

// Internal model implementation interface. Concrete models (e.g. QwenModel)
// derive from this and implement the graph-building surface. This is the
// engine-facing counterpart of the public Model facade and is never exposed.
class ModelImpl {
public:
  explicit ModelImpl(ModelLoader *loader, bool cpu_only = false);
  virtual ~ModelImpl();

  // Graph-building surface (implemented by concrete models).
  virtual ggml_tensor *build_graph(ggml_context *ctx, ggml_cgraph *gf,
                                   KVCache *cache) = 0;
  virtual void summary() = 0;

  void create_input_tensors(ggml_context *ctx, size_t n_tokens,
                            size_t total_filled);
  std::vector<ggml_tensor *> get_input_tensors();
  void set_inputs(const model_input_t &input);

  // Metadata passthrough.
  zinferlm::model_info_t info() const;
  zinferlm::tokenizer_info_t tokenizer_info() const;
  std::vector<zinferlm::tensor_info_t> tensor_info() const;
  zinferlm::model_config_t config() const;

  // Backend access.
  ggml_backend *get_backend();
  std::vector<ggml_backend *> backends() const;
  static enum ggml_backend_dev_type get_backend_type();
  bool use_gpu() const;

  // Tensor loading.
  void create_tensor_structs();
  void mark_tensors_external();
  void load_tensors();

  // All weight tensor structs (no data access implied).
  std::vector<ggml_tensor *> tensors() const;
  // Bytes of weight data, summed over the tensor structs.
  uint64_t size() const { return model_size_; }
  // Host arena used for the tensor metadata (as reserved by the ctor).
  size_t tensor_metadata_size() const;

protected:
  ggml_tensor *get_tensor_(std::string name);

  ggml_context_ptr tensor_ctx_;
  std::unordered_map<std::string, ggml_tensor *> tensor_map_;
  std::unique_ptr<ModelLoader> loader_;
  std::vector<ggml_backend_ptr> backends_;
  std::vector<ggml_tensor *> input_tensors_;
  uint64_t model_size_ = 0;
  model_metrics_t metrics_;

  void build_mask(std::vector<ggml_fp16_t> &mask, const int filled,
                  const int size);
  void init_metrics();

private:
  ggml_backend_buffer_ptr tensor_buffer_;
  bool cpu_only_ = false;
};

} // namespace internal
} // namespace zinferlm
