#include <cmath>
#include <cstdint>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

#include "engine/tensors.h"
#include "ggml.h"
#include "layers/layers.h"
#include "model.h"

QwenModel::QwenModel(zinferlm::ModelLoader *l) : zinferlm::Model(l) {}

void QwenModel::create_input_tensors(ggml_context *ctx, uint32_t n_tokens,
                                     uint32_t max_len) {
  ggml_tensor *input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  ggml_set_input(input);
  ggml_set_name(input, "input");

  ggml_tensor *positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  ggml_set_input(positions);
  ggml_set_name(positions, "positions");

  ggml_tensor *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, max_len, n_tokens);
  ggml_set_input(mask);
  ggml_set_name(mask, "mask");

  input_tensors_ = {input, positions, mask};
  max_context_len_ = max_len;
}

std::vector<ggml_tensor *> QwenModel::get_input_tensors() {
  return input_tensors_;
}

void QwenModel::set_inputs(const std::vector<int32_t> &tokens,
                           int past_tokens) {
  ggml_tensor *input = input_tensors_[0];
  ggml_tensor *positions = input_tensors_[1];
  ggml_tensor *mask = input_tensors_[2];

  ggml_backend_tensor_set(input, tokens.data(), 0,
                          tokens.size() * sizeof(int32_t));

  std::vector<int32_t> pos(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) {
    pos[i] = past_tokens + (int32_t)i;
  }
  ggml_backend_tensor_set(positions, pos.data(), 0,
                          pos.size() * sizeof(int32_t));

  std::vector<ggml_fp16_t> mask_data((size_t)max_context_len_ * tokens.size());
  for (size_t q = 0; q < tokens.size(); q++) {
    int pos_q = past_tokens + (int)q;
    for (uint32_t k = 0; k < max_context_len_; k++) {
      float v = ((int)k <= pos_q) ? 0.0f : -INFINITY;
      mask_data[q * max_context_len_ + k] = ggml_fp32_to_fp16(v);
    }
  }
  ggml_backend_tensor_set(mask, mask_data.data(), 0,
                          mask_data.size() * sizeof(ggml_fp16_t));
}

ggml_tensor *QwenModel::build_graph(ggml_context *ctx, ggml_cgraph *gf,
                                    KVCache *cache) {
  uint32_t nblocks = loader_->model_config().n_blocks;

  ggml_tensor *input = input_tensors_[0];
  ggml_tensor *positions = input_tensors_[1];
  ggml_tensor *mask = input_tensors_[2];

  ggml_tensor *cur = input;

  TokenEmbedding emb = create_embedding_layer_(ctx, "token_embd");
  cur = emb(cur);

  for (uint32_t i = 0; i < nblocks; i++) {
    std::string blk = "blk." + std::to_string(i);

    GroupedAttentionHead attn = create_attention_layer_(
        ctx, gf, positions, mask, blk + ".attn", i, cache);
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
    ggml_tensor *mask, std::string layer_name, int block_id, KVCache *cache) {
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
                                          .mask = mask};
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
