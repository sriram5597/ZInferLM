#include <cstdint>
#include <cstring>
#include <ggml-cpp.h>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

#include "engine/tensors.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "layers/layers.h"
#include "model.h"
#include "model_loader/loader.h"
QwenModel::QwenModel(std::unique_ptr<ModelLoader> l) : loader_(std::move(l)) {}

zinferlm::model_info_t QwenModel::info() const { return loader_->info(); }

zinferlm::tokenizer_info_t QwenModel::tokenizer_info() const {
  return loader_->tokenizer_info();
}

std::vector<zinferlm::tensor_info_t> QwenModel::tensor_info() const {
  return loader_->tensor_info();
}

zinferlm::model_config_t QwenModel::config() const {
  return loader_->model_config();
}

std::vector<std::unique_ptr<Layer>> QwenModel::create_layers(ggml_context *ctx, int past_tokens, int len, KVCache *cache) {
  uint32_t nblocks = loader_->model_config().n_blocks;
  std::vector<std::unique_ptr<Layer>> layers;

  Layer* emb = create_embedding_layer_(ctx, "token_embd");
  layers.push_back(std::unique_ptr<Layer>(emb));

  for (uint32_t i = 0; i < nblocks; i++) {
    Layer* attn = create_attention_layer_(
        ctx, "blk." + std::to_string(i) + ".attn", i, past_tokens, len, cache);
    Layer* ffn = create_ffn_layer_(ctx, "blk." + std::to_string(i) + ".ffn", i);
    layers.push_back(std::unique_ptr<Layer>(attn));
    layers.push_back(std::unique_ptr<Layer>(ffn));
  }

  Layer* out = create_output_layer_(ctx, "output");
  layers.push_back(std::unique_ptr<Layer>(out));

  return layers;
}

Layer *QwenModel::create_embedding_layer_(ggml_context *ctx,
                                          std::string layer_name) {
  zinferlm::tensor_info_t info = loader_->tensor_info(layer_name + ".weight");
  token_embedding_params_t params = {.ctx = ctx,
                                     .emb_w = create_tensor(ctx, info)};
  return new TokenEmbedding(params);
}

Layer *QwenModel::create_output_layer_(ggml_context *ctx,
                                       std::string layer_name) {
  zinferlm::model_config_t config = loader_->model_config();
  zinferlm::tensor_info_t info = loader_->tensor_info(layer_name + ".weight");
  zinferlm::tensor_info_t norm_info =
      loader_->tensor_info(layer_name + "_norm.weight");
  token_unembedding_params_t uemb_params = {.ctx = ctx,
                                            .unemb_w = create_tensor(ctx, info),
                                            .norm_gamma =
                                                create_tensor(ctx, norm_info),
                                            .norm_eps = config.rms_eps};
  return new TokenUnembedding(uemb_params);
}

Layer *QwenModel::create_attention_layer_(ggml_context *ctx,
                                          std::string layer_name, int block_id,
                                          int past_tokens, int seq_len,
                                          KVCache *cache) {
  zinferlm::model_config_t config = loader_->model_config();
  zinferlm::tensor_info_t q_w = loader_->tensor_info(layer_name + "_q.weight");
  zinferlm::tensor_info_t q_b = loader_->tensor_info(layer_name + "_q.bias");
  zinferlm::tensor_info_t k_w = loader_->tensor_info(layer_name + "_k.weight");
  zinferlm::tensor_info_t k_b = loader_->tensor_info(layer_name + "_k.bias");
  zinferlm::tensor_info_t v_w = loader_->tensor_info(layer_name + "_v.weight");
  zinferlm::tensor_info_t v_b = loader_->tensor_info(layer_name + "_v.bias");
  zinferlm::tensor_info_t out_w =
      loader_->tensor_info(layer_name + "_output.weight");
  zinferlm::tensor_info_t norm_info =
      loader_->tensor_info(layer_name + "_norm.weight");
  grouped_attn_head_params attn_params = {
      .block_id = block_id,
      .ctx = ctx,
      .q_w = create_tensor(ctx, q_w),
      .k_w = create_tensor(ctx, k_w),
      .q_b = create_tensor(ctx, q_b),
      .k_b = create_tensor(ctx, k_b),
      .v_w = create_tensor(ctx, v_w),
      .v_b = create_tensor(ctx, v_b),
      .out_w = create_tensor(ctx, out_w),
      .rope_freq_base = config.rope_freq_base,
      .apply_rope = true,
      .n_heads = config.nheads,
      .n_kv = config.nkv,
      .d_model = config.embedding_dim,
      .past_tokens = past_tokens,
      .len = seq_len,
      .residual = true,
      .norm_gamma = create_tensor(ctx, norm_info),
      .norm_eps = config.rms_eps,
      .debug = debug_,
      .cache=cache
  };
  return new GroupedAttentionHead(attn_params);
}

Layer *QwenModel::create_ffn_layer_(ggml_context *ctx, std::string layer_name,
                                    int block_id) {
  zinferlm::model_config_t config = loader_->model_config();
  zinferlm::tensor_info_t up_w =
      loader_->tensor_info(layer_name + "_up.weight");
  zinferlm::tensor_info_t gate =
      loader_->tensor_info(layer_name + "_gate.weight");
  zinferlm::tensor_info_t down =
      loader_->tensor_info(layer_name + "_down.weight");
  zinferlm::tensor_info_t norm_info =
      loader_->tensor_info(layer_name + "_norm.weight");
  swiglu_params_t sparams = {
      .ctx = ctx,
      .w_gate = create_tensor(ctx, gate),
      .w_down = create_tensor(ctx, down),
      .w_up = create_tensor(ctx, up_w),
      .residual = true,
      .norm_gamma = create_tensor(ctx, norm_info),
      .norm_eps = config.rms_eps,
  };
  return new SwigLU(sparams);
}

void QwenModel::summary() {
  // ggml_context_ptr ctx = init_engine(loader_->get_tensor_count() * 48 + 128);
  // Graph graph(ctx.get());
  // std::unique_ptr<KVCache> cache =
  // std::make_unique<KVCache>(ggml_backend_get_default_buffer_type(graph.get_backend()),
  // 512,
  //               nblocks, config.nkv, config.embedding_dim / config.nheads,
  //               GGML_TYPE_F16);
  // std::vector<Layer *> layers = build_graph_(ctx.get(), 0, 1, cache.get());
  // ggml_cgraph *gf = graph.build(1);
  // ggml_graph_print(gf);
}
