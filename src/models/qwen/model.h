#include <cstdint>
#include <ggml-cpp.h>
#include <ggml.h>
#include <memory>
#include <utility>
#include <zinferlm/models.h>

#include "layers/layers.h"
#include "kv_cache/cache.h"
#include "model_loader/loader.h"

class QwenModel : public zinferlm::Model {
private:
  std::unique_ptr<ModelLoader> loader_;
  ggml_context_ptr ctx_;
  Layer* create_embedding_layer_(ggml_context *ctx, std::string layer_name);
  Layer* create_output_layer_(ggml_context* ctx, std::string layer_name);
  Layer* create_attention_layer_(ggml_context* ctx, std::string layer_name, int block_id, int past_tokens, int seq_len, KVCache* cache);
  Layer* create_ffn_layer_(ggml_context* ctx, std::string layer_name, int block_id);

public:
  QwenModel(std::unique_ptr<ModelLoader> f);
  zinferlm::model_info_t info() const override;
  zinferlm::tokenizer_info_t tokenizer_info() const override;
  std::vector<zinferlm::tensor_info_t> tensor_info() const override;
  zinferlm::model_config_t config() const override;
  std::vector<std::unique_ptr<Layer>> create_layers(ggml_context *ctx, int past_tokens, int len, KVCache* cache) override;
  void summary() override;
};
