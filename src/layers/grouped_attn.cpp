#include <cmath>
#include <cstddef>
#include <cstring>
#include <ggml-backend.h>
#include <ggml.h>

#include "layers.h"

GroupedAttentionHead::GroupedAttentionHead(grouped_attn_head_params params)
    : Layer(params.ctx, params.residual, params.norm_gamma, params.norm_eps),
      q_w_(params.q_w), q_b_(params.q_b), k_w_(params.k_w),
      k_b_(params.k_b), v_w_(params.v_w), v_b_(params.v_b),
      out_w_(params.out_w), rope_freq_base_(params.rope_freq_base),
      apply_rope_(params.apply_rope), n_heads_(params.n_heads),
      n_kv_(params.n_kv), past_tokens_(params.past_tokens), len_(params.len) {
  name = "GroupedAttention";
}

ggml_tensor *GroupedAttentionHead::forward(ggml_tensor *x) const {
  ggml_tensor *Q = ggml_mul_mat(ctx_, q_w_, x);
  if (q_b_ != nullptr) {
    Q = ggml_add(ctx_, Q, q_b_);
  }
  ggml_tensor *K = ggml_mul_mat(ctx_, k_w_, x);
  if (k_b_ != nullptr) {
    K = ggml_add(ctx_, K, k_b_);
  }
  ggml_tensor *V = ggml_mul_mat(ctx_, v_w_, x);
  if (v_b_ != nullptr) {
    V = ggml_add(ctx_, V, v_b_);
  }
  int d_head = x->ne[0] / n_heads_;
  ggml_tensor *Q_transformed =
      ggml_reshape_4d(ctx_, Q, d_head, n_heads_, len_, 1);
  ggml_tensor *K_transformed = ggml_reshape_4d(ctx_, K, d_head, n_kv_, len_, 1);
  ggml_tensor *V_transformed = ggml_reshape_4d(ctx_, V, d_head, n_kv_, len_, 1);

  if (apply_rope_) {
    ggml_tensor *positions =
        ggml_cast(ctx_, ggml_arange(ctx_, past_tokens_, past_tokens_ + len_, 1),
                  GGML_TYPE_I32);
    Q_transformed =
        ggml_rope_ext(ctx_, Q_transformed, positions, NULL, d_head,
                      GGML_ROPE_TYPE_NEOX, 0, rope_freq_base_, 1.0f, 0.0f,
                      1.0f, 0.0f, 0.0f);
    K_transformed =
        ggml_rope_ext(ctx_, K_transformed, positions, NULL, d_head,
                      GGML_ROPE_TYPE_NEOX, 0, rope_freq_base_, 1.0f, 0.0f,
                      1.0f, 0.0f, 0.0f);
  }

  Q_transformed = ggml_permute(ctx_, Q_transformed, 0, 2, 1, 3);
  K_transformed = ggml_permute(ctx_, K_transformed, 0, 2, 1, 3);
  V_transformed = ggml_permute(ctx_, V_transformed, 0, 2, 1, 3);

  ggml_tensor *KQ = ggml_mul_mat(
      ctx_, ggml_repeat(ctx_, K_transformed, Q_transformed), Q_transformed);
  ggml_mul_mat_set_prec(KQ, GGML_PREC_F32);
  const float scale = 1.0f / sqrtf((float)d_head);
  KQ = ggml_scale(ctx_, KQ, scale);
  KQ = ggml_diag_mask_inf(ctx_, KQ, past_tokens_);
  V_transformed = ggml_repeat(ctx_, V_transformed, Q_transformed);
  V_transformed = ggml_permute(ctx_, V_transformed, 1, 0, 2, 3);

  ggml_tensor *scaled_attn = ggml_mul_mat(ctx_, ggml_cont(ctx_, V_transformed),
                                          ggml_soft_max(ctx_, KQ));
  ggml_tensor *attn_trans = ggml_permute(ctx_, scaled_attn, 0, 2, 1, 3);
  attn_trans = ggml_cont(ctx_, attn_trans);
  attn_trans = ggml_reshape_2d(ctx_, attn_trans, x->ne[0], x->ne[1]);

  ggml_tensor *attn_out = ggml_mul_mat(ctx_, out_w_, attn_trans);

  return attn_out;
}
