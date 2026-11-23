#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

#include "engine/tensors.h"
#include "layers/layers.h"
#include "model.h"

namespace zinferlm {
namespace internal {

QwenModel::QwenModel(ModelLoader *l, bool cpu_only)
    : ModelImpl(l, cpu_only) {}

ggml_tensor *QwenModel::build_graph(ggml_context *ctx, ggml_cgraph *gf,
                                    KVCache *cache) {
  uint32_t nblocks = loader_->model_config().n_blocks;

  ggml_tensor *input = input_tensors_[0];
  ggml_tensor *positions = input_tensors_[1];
  ggml_tensor *mask = input_tensors_[2];
  ggml_tensor *lookup_rows = input_tensors_[3];
  ggml_tensor *write_rows = input_tensors_[4];

  ggml_tensor *cur = input;

  TokenEmbedding emb = create_embedding_layer_(ctx, "token_embd");
  cur = emb(cur);

  for (uint32_t i = 0; i < nblocks; i++) {
    std::string blk = "blk." + std::to_string(i);

    GroupedAttentionHead attn =
        create_attention_layer_(ctx, gf, positions, mask, lookup_rows,
                                write_rows, blk + ".attn", i, cache);
    cur = attn(cur);

    SwigLU ffn = create_ffn_layer_(ctx, blk + ".ffn", i);
    cur = ffn(cur);
  }

  TokenUnembedding out = create_output_layer_(ctx, "output");
  cur = out(cur);

  return cur;
}

TokenEmbedding QwenModel::create_embedding_layer_(ggml_context *ctx,
                                                  std::string layer_name) {
  token_embedding_params_t params = {
      .ctx = ctx, .emb_w = get_tensor_(layer_name + ".weight")};
  return TokenEmbedding(params);
}

TokenUnembedding QwenModel::create_output_layer_(ggml_context *ctx,
                                                 std::string layer_name) {
  zinferlm::model_config_t config = loader_->model_config();
  token_unembedding_params_t uemb_params = {
      .ctx = ctx,
      .unemb_w = get_tensor_(layer_name + ".weight"),
      .norm_gamma = get_tensor_(layer_name + "_norm.weight"),
      .norm_eps = config.rms_eps};
  return TokenUnembedding(uemb_params);
}

GroupedAttentionHead QwenModel::create_attention_layer_(
    ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *positions,
    ggml_tensor *mask, ggml_tensor *lookup_rows, ggml_tensor *write_rows,
    std::string layer_name, int block_id, KVCache *cache) {
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
                                          .residual = true,
                                          .norm_gamma = norm_info,
                                          .norm_eps = config.rms_eps,
                                          .cache = cache,
                                          .gf = gf,
                                          .positions = positions,
                                          .mask = mask,
                                          .lookup_rows = lookup_rows,
                                          .write_rows = write_rows};
  return GroupedAttentionHead(attn_params);
}

SwigLU QwenModel::create_ffn_layer_(ggml_context *ctx, std::string layer_name,
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
  return SwigLU(sparams);
}

void QwenModel::summary() {}

} // namespace internal
} // namespace zinferlm
