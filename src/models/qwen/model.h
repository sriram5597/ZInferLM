#include <cstdint>
#include <ggml-cpp.h>
#include <ggml.h>
#include <memory>
#include <utility>
#include <zinferlm/models.h>

#include "layers/layers.h"
#include "model_loader/loader.h"

class QwenModel : public zinferlm::Model {
private:
  std::unique_ptr<ModelLoader> loader_;
  ggml_context_ptr ctx_;
  std::vector<std::unique_ptr<Layer>> build_graph_(ggml_context *ctx,
                                                   int past_tokens, int len);
  std::vector<std::pair<Layers, std::string>> pre_attn_blocks_ = {
      {Layers::TOKEN_EMBEDDING, "token_embd"}};
  std::vector<std::pair<Layers, std::string>> attn_blocks_ = {
      {Layers::GROUPED_ATTN, "attn"},
      {Layers::SWIGLU, "ffn"}};
  std::vector<std::pair<Layers, std::string>> post_attn_blocks_ = {
      {Layers::TOKEN_UNEMBEDDING, "output"}};
  Layer *create_layer_(ggml_context *ctx, std::string layer_name,
                       Layers layer_type, int past_tokens, int len);

public:
  QwenModel(std::unique_ptr<ModelLoader> f);
  zinferlm::model_info_t info() const override;
  zinferlm::tokenizer_info_t tokenizer_info() const override;
  std::vector<zinferlm::tensor_info_t> tensor_info() const override;
  std::vector<float> invoke(std::vector<uint32_t> tokens) override;
  void summary() override;
  Block *create_attn_block_(ggml_context *ctx, int block_id, int past_tokens,
                            int len);
};
