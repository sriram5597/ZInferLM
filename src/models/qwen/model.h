#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <ggml-cpp.h>
#include <ggml.h>
#include <zinferlm/models.h>

#include "../model_impl.h"
#include "layers/layers.h"

namespace zinferlm {
namespace internal {

class QwenModel : public ModelImpl {
private:
  uint32_t max_context_len_ = 0;

  TokenEmbedding create_embedding_layer_(ggml_context *ctx,
                                         std::string layer_name);
  TokenUnembedding create_output_layer_(ggml_context *ctx,
                                        std::string layer_name);
  GroupedAttentionHead
  create_attention_layer_(ggml_context *ctx, ggml_cgraph *gf,
                          ggml_tensor *positions, ggml_tensor *mask,
                          ggml_tensor *lookup_rows, ggml_tensor* write_rows, std::string layer_name,
                          int block_id, KVCache *cache);
  SwigLU create_ffn_layer_(ggml_context *ctx, std::string layer_name,
                           int block_id);

public:
  QwenModel(ModelLoader *f);
  ggml_tensor *build_graph(ggml_context *ctx, ggml_cgraph *gf,
                           KVCache *cache) override;
  void summary() override;
};

} // namespace internal
} // namespace zinferlm
