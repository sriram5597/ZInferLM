#include <cstdint>
#include <ggml-cpp.h>
#include <memory>
#include <string>
#include <utility>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

#include "engine/tensors.h"
#include "ggml.h"
#include "layers/layers.h"
#include "model.h"

QwenModel::QwenModel(zinferlm::ModelLoader *l) : zinferlm::Model(l) {}

std::vector<std::unique_ptr<Layer>> QwenModel::create_layers(ggml_context *ctx,
                                                             int past_tokens,
                                                             int len,
                                                             KVCache *cache) {
  uint32_t nblocks = loader_->model_config().n_blocks;
  std::vector<std::unique_ptr<Layer>> layers;

  Layer *emb = create_embedding_layer_(ctx, "token_embd");
  layers.push_back(std::unique_ptr<Layer>(emb));

  for (uint32_t i = 0; i < nblocks; i++) {
    Layer *attn = create_attention_layer_(
        ctx, "blk." + std::to_string(i) + ".attn", i, past_tokens, len, cache);
    Layer *ffn = create_ffn_layer_(ctx, "blk." + std::to_string(i) + ".ffn", i);
    layers.push_back(std::unique_ptr<Layer>(attn));
    layers.push_back(std::unique_ptr<Layer>(ffn));
  }

  Layer *out = create_output_layer_(ctx, "output");
  layers.push_back(std::unique_ptr<Layer>(out));

  return layers;
}

Layer *QwenModel::create_embedding_layer_(ggml_context *ctx,
                                          std::string layer_name) {
  token_embedding_params_t params = {
      .ctx = ctx, .emb_w = get_tensor_(layer_name + ".weight")};
  return new TokenEmbedding(params);
}

Layer *QwenModel::create_output_layer_(ggml_context *ctx,
                                       std::string layer_name) {
  zinferlm::model_config_t config = loader_->model_config();
  token_unembedding_params_t uemb_params = {
      .ctx = ctx,
      .unemb_w = get_tensor_(layer_name + ".weight"),
      .norm_gamma = get_tensor_(layer_name + "_norm.weight"),
      .norm_eps = config.rms_eps};
  return new TokenUnembedding(uemb_params);
}

Layer *QwenModel::create_attention_layer_(ggml_context *ctx,
                                          std::string layer_name, int block_id,
                                          int past_tokens, int seq_len,
                                          KVCache *cache) {
  zinferlm::model_config_t config = loader_->model_config();
  ggml_tensor *q_w = get_tensor_(layer_name + "_q.weight");
  ggml_tensor *q_b = get_tensor_(layer_name + "_q.bias");
  ggml_tensor *k_w = get_tensor_(layer_name + "_k.weight");
  ggml_tensor *k_b = get_tensor_(layer_name + "_k.bias");
  ggml_tensor *v_w = get_tensor_(layer_name + "_v.weight");
  ggml_tensor *v_b = get_tensor_(layer_name + "_v.bias");
  ggml_tensor *out_w = get_tensor_(layer_name + "_output.weight");
  ggml_tensor *norm_info = get_tensor_(layer_name + "_norm.weight");
  grouped_attn_head_params attn_params = {.block_id = block_id,
                                          .ctx = ctx,
                                          .q_w = q_w,
                                          .k_w = k_w,
                                          .q_b = q_b,
                                          .k_b = k_b,
                                          .v_w = v_w,
                                          .v_b = v_b,
                                          .out_w = out_w,
                                          .rope_freq_base =
                                              config.rope_freq_base,
                                          .apply_rope = true,
                                          .n_heads = config.nheads,
                                          .n_kv = config.nkv,
                                          .d_model = config.embedding_dim,
                                          .past_tokens = past_tokens,
                                          .len = seq_len,
                                          .residual = true,
                                          .norm_gamma = norm_info,
                                          .norm_eps = config.rms_eps,
                                          .debug = debug_,
                                          .cache = cache};
  return new GroupedAttentionHead(attn_params);
}

Layer *QwenModel::create_ffn_layer_(ggml_context *ctx, std::string layer_name,
                                    int block_id) {
  zinferlm::model_config_t config = loader_->model_config();
  ggml_tensor *up_w = get_tensor_(layer_name + "_up.weight");
  ggml_tensor *gate = get_tensor_(layer_name + "_gate.weight");
  ggml_tensor *down = get_tensor_(layer_name + "_down.weight");
  ggml_tensor *norm_info = get_tensor_(layer_name + "_norm.weight");
  swiglu_params_t sparams = {
      .ctx = ctx,
      .w_gate = gate,
      .w_down = down,
      .w_up = up_w,
      .residual = true,
      .norm_gamma = norm_info,
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
