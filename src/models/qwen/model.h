#pragma once

#include <cstdint>
#include <ggml-cpp.h>
#include <ggml.h>
#include <memory>
#include <string>
#include <vector>
#include <zinferlm/models.h>

#include "layers/layers.h"

class QwenModel : public zinferlm::Model {
private:
  std::vector<ggml_tensor *> input_tensors_;
  uint32_t max_context_len_ = 0;

  TokenEmbedding create_embedding_layer_(ggml_context *ctx,
                                         std::string layer_name);
  TokenUnembedding create_output_layer_(ggml_context *ctx,
                                        std::string layer_name);
  GroupedAttentionHead
  create_attention_layer_(ggml_context *ctx, ggml_cgraph *gf,
                          ggml_tensor *positions, ggml_tensor *mask,
                          std::string layer_name, int block_id, KVCache *cache);
  SwigLU create_ffn_layer_(ggml_context *ctx, std::string layer_name,
                           int block_id);

public:
  QwenModel(zinferlm::ModelLoader *f);
  ggml_tensor *build_graph(ggml_context *ctx, ggml_cgraph *gf,
                           KVCache *cache) override;
  void create_input_tensors(ggml_context *ctx, uint32_t n_tokens,
                            uint32_t max_len) override;
  std::vector<ggml_tensor *> get_input_tensors() override;
  void set_inputs(const std::vector<int32_t> &tokens, int past_tokens) override;
  void summary() override;
};
