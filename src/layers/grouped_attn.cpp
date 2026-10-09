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
      layer_index_(params.layer_index), cache_(params.cache), gf_(params.gf),
      positions_(params.positions), mask_(params.mask) {
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
  const int64_t len = x->ne[1];

  // Reshape to 3D: [d_head, n_heads/n_kv, seq_len]
  ggml_tensor *Q_cur = ggml_reshape_3d(ctx_, Q, d_head, n_heads_, len);
  ggml_tensor *K_cur = ggml_reshape_3d(ctx_, K, d_head, n_kv_, len);
  ggml_tensor *V_cur = ggml_reshape_3d(ctx_, V, d_head, n_kv_, len);

  // Apply RoPE
  if (apply_rope_) {
    Q_cur = ggml_rope_ext(ctx_, Q_cur, positions_, NULL, d_head,
                          GGML_ROPE_TYPE_NEOX, 0, rope_freq_base_, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
    K_cur = ggml_rope_ext(ctx_, K_cur, positions_, NULL, d_head,
                          GGML_ROPE_TYPE_NEOX, 0, rope_freq_base_, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
  }
  ggml_set_name(Q_cur, "Qcur_rope");
  ggml_set_name(K_cur, "Kcur_rope");
  Q_cur = ggml_cont(ctx_, ggml_permute(ctx_, Q_cur, 0, 2, 1, 3));

  cache_->set_graph(gf_);
  cache_->write_kv(ctx_, block_id_, K_cur, V_cur, positions_);
  layer_kv_cache_t cached_entry = cache_->get_slice(ctx_, block_id_);
  K_cur = cached_entry.K;
  V_cur = cached_entry.V;

  ggml_tensor *KQV = ggml_flash_attn_ext(ctx_, Q_cur, K_cur, V_cur, mask_,
                                         1.0f / sqrtf(d_head), 0.0f, 0.0f);
  ggml_set_name(KQV, "KQV");
  KQV = ggml_reshape_2d(ctx_, KQV, x->ne[0], x->ne[1]);

  // Output projection
  ggml_tensor *attn_out = ggml_mul_mat(ctx_, out_w_, KQV);
  ggml_set_name(attn_out, "attn-out");

  return attn_out;
}
