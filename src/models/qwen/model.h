#include <ggml-cpp.h>
#include <ggml.h>
#include <memory>
#include <zinferlm/models.h>

#include "layers/layers.h"

class QwenModel : public zinferlm::Model {
private:
  ggml_context_ptr ctx_;
  Layer *create_embedding_layer_(ggml_context* ctx, std::string layer_name);
  Layer *create_output_layer_(ggml_context* ctx, std::string layer_name);
  Layer *create_attention_layer_(ggml_context* ctx, std::string layer_name, int block_id,
                                 int past_tokens, int seq_len, KVCache *cache);
  Layer *create_ffn_layer_(ggml_context* ctx, std::string layer_name, int block_id);

public:
  QwenModel(zinferlm::ModelLoader *f);
  std::vector<std::unique_ptr<Layer>> create_layers(ggml_context* ctx, int past_tokens, int len,
                                                    KVCache *cache) override;
  void summary() override;
};
