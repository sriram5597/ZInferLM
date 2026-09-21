#include <cmath>
#include <cstddef>
#include <cstring>
#include <ggml-backend.h>
#include <ggml.h>
#include <iostream>

#include "layers.h"

GroupedAttentionHead::GroupedAttentionHead(grouped_attn_head_params params)
    : Layer(params.ctx, params.residual, params.norm_gamma, params.norm_eps),
      block_id_(params.block_id), q_w_(params.q_w), q_b_(params.q_b),
      k_w_(params.k_w), k_b_(params.k_b), v_w_(params.v_w), v_b_(params.v_b),
      out_w_(params.out_w), out_b_(params.out_b),
      rope_freq_base_(params.rope_freq_base), apply_rope_(params.apply_rope),
      n_heads_(params.n_heads), n_kv_(params.n_kv),
      past_tokens_(params.past_tokens), len_(params.len), debug_(params.debug),
      layer_index_(params.layer_index), cache_(params.cache) {
  name = "GroupedAttention";
}

ggml_tensor *GroupedAttentionHead::forward(ggml_tensor *x) const {
  // Linear projections
  ggml_tensor *Q = ggml_mul_mat(ctx_, q_w_, x);
  ggml_set_name(Q, "Qcur");
  if (q_b_ != nullptr) {
    Q = ggml_add(ctx_, Q, q_b_);
    ggml_set_name(Q, "Qcur");
  }
  ggml_tensor *K = ggml_mul_mat(ctx_, k_w_, x);
  ggml_set_name(K, "Kcur");
  if (k_b_ != nullptr) {
    K = ggml_add(ctx_, K, k_b_);
    ggml_set_name(K, "Kcur");
  }

  ggml_tensor *V = ggml_mul_mat(ctx_, v_w_, x);
  ggml_set_name(V, "Vcur");
  if (v_b_ != nullptr) {
    V = ggml_add(ctx_, V, v_b_);
    ggml_set_name(V, "Vcur");
  }

  int d_head = x->ne[0] / n_heads_;

  // Reshape to 3D: [d_head, n_heads/n_kv, seq_len]
  ggml_tensor *Q_cur = ggml_reshape_3d(ctx_, Q, d_head, n_heads_, len_);
  ggml_tensor *K_cur = ggml_reshape_3d(ctx_, K, d_head, n_kv_, len_);
  ggml_tensor *V_cur = ggml_reshape_3d(ctx_, V, d_head, n_kv_, len_);

  // Apply RoPE
  if (apply_rope_) {
    ggml_tensor *positions =
        ggml_cast(ctx_, ggml_arange(ctx_, past_tokens_, past_tokens_ + len_, 1),
                  GGML_TYPE_I32);
    Q_cur =
        ggml_rope_ext(ctx_, Q_cur, positions, NULL, d_head, GGML_ROPE_TYPE_NEOX,
                      0, rope_freq_base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K_cur =
        ggml_rope_ext(ctx_, K_cur, positions, NULL, d_head, GGML_ROPE_TYPE_NEOX,
                      0, rope_freq_base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
  }
  ggml_set_name(Q_cur, "Qcur_rope");
  ggml_set_name(K_cur, "Kcur_rope");
  K_cur = ggml_cont(ctx_, ggml_permute(ctx_, K_cur, 0, 2, 1, 3));
  V_cur = ggml_cont(ctx_, ggml_permute(ctx_, V_cur, 0, 2, 1, 3));
  Q_cur = ggml_cont(ctx_, ggml_permute(ctx_, Q_cur, 0, 2, 1, 3));

  cache_->set_graph(gf_);
  cache_->concat(ctx_, block_id_, len_, K_cur, V_cur);
  layer_kv_cache_t cached_entry = cache_->get_slice(ctx_, block_id_);
  K_cur = cached_entry.K;
  V_cur = cached_entry.V;

  struct ggml_tensor *mask_f32 =
      ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, past_tokens_ + len_, past_tokens_ + len_);
  mask_f32 = ggml_scale(ctx_, mask_f32, 0.0f);
  mask_f32 = ggml_diag_mask_inf(ctx_, mask_f32, past_tokens_);
  ggml_set_name(mask_f32, "mask_f32");
  struct ggml_tensor *mask = ggml_cast(ctx_, mask_f32, GGML_TYPE_F16);
  ggml_set_name(mask, "mask");
  ggml_tensor *KQV = ggml_flash_attn_ext(ctx_, Q_cur, K_cur, V_cur, mask,
                                         1.0f / sqrtf(d_head), 0.0f, 0.0f);
  ggml_set_name(KQV, "KQV");

  // // Permute Q and K to [d_head, seq_len, n_heads/n_kv] for attention
  // // computation
  // Q_cur = ggml_permute(ctx_, Q_cur, 0, 2, 1, 3);
  // K_cur = ggml_permute(ctx_, K_cur, 0, 2, 1, 3);
  //
  // // Permute V to [n_heads/n_kv, seq_len, d_head] for later
  // multiplication V_cur = ggml_permute(ctx_, V_cur, 0, 2, 1, 3);
  //
  // if (n_kv_ < n_heads_) {
  //   K_cur = ggml_repeat(ctx_, K_cur, Q_cur);
  //   V_cur = ggml_repeat(ctx_, V_cur, Q_cur);
  // }
  //
  // ggml_tensor *KQ = ggml_mul_mat(ctx_, K_cur, Q_cur);
  //
  // // Scale by 1/sqrt(d_head)
  // const float scale = 1.0f / sqrtf((float)d_head);
  // KQ = ggml_scale(ctx_, KQ, scale);
  // ggml_set_name(KQ, "KQ_scores");
  //
  // // Apply causal mask
  // KQ = ggml_diag_mask_inf(ctx_, KQ, past_tokens_);
  // ggml_set_name(KQ, "KQ_masked");
  //
  // // Apply softmax to get attention weights
  // ggml_tensor *KQ_soft = ggml_soft_max(ctx_, KQ);
  // ggml_set_name(KQ_soft, "kq-softmax");
  // // Compute attention output: V @ softmax(KQ)
  // // V_cur: [d_head, seq_len, n_heads], KQ_soft: [seq_len, seq_len,
  // n_heads]
  // // Result: [d_head, seq_len, n_heads]
  // V_cur = ggml_cont(ctx_, ggml_permute(ctx_, V_cur, 1, 0, 2, 3));
  // ggml_tensor *KQV = ggml_mul_mat(ctx_, V_cur, KQ_soft);
  // ggml_set_name(KQV, "kqv");
  //
  // // Permute back to [d_head, n_heads, seq_len]
  KQV = ggml_reshape_2d(ctx_, KQV, x->ne[0], x->ne[1]);

  // Output projection
  ggml_tensor *attn_out = ggml_mul_mat(ctx_, out_w_, KQV);
  ggml_set_name(attn_out, "attn-out");

  return attn_out;
}
