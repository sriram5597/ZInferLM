#include <cmath>
#include <ggml.h>

#include "layers.h"

GroupedAttentionHead::GroupedAttentionHead(grouped_attn_head_params params)
    : ctx_(params.ctx), q_w_(params.q_w), q_b_(params.q_b), k_w_(params.k_w),
      k_b_(params.k_b), v_w_(params.v_w), v_b_(params.v_b),
      out_w_(params.out_w), out_b_(params.out_b),
      rope_freq_base_(params.rope_freq_base), apply_rope_(params.apply_rope),
      n_heads_(params.n_heads), n_kv_(params.n_kv) {
  name = "GroupedAttention";
}

ggml_tensor *GroupedAttentionHead::operator()(ggml_tensor *x) const {
  ggml_tensor *Q = ggml_mul_mat(ctx_, q_w_, x);
  if (q_b_ != nullptr) {
    Q = ggml_add(ctx_, q_b_, Q);
  }
  ggml_tensor *K = ggml_mul_mat(ctx_, k_w_, x);
  if (k_b_ != nullptr) {
    K = ggml_add(ctx_, k_b_, K);
  }
  ggml_tensor *V = ggml_mul_mat(ctx_, v_w_, x);
  if (v_b_ != nullptr) {
    V = ggml_add(ctx_, v_b_, V);
  }
  if (apply_rope_) {
    Q = ggml_rope_ext(ctx_, // 1.  Computation context
                      Q,    // 2.  Input tensor to rotate (Query or Key)
                      NULL, // 3.  Custom position tensor (optional)
                      0,    // 4.  Sequence position offset (n_past)
                      0,    // 5.  Number of rotated dimensions (n_dims)
                      0,    // 6.  RoPE mode / style flags
                      0,    // 7.  Original context training length (n_ctx_orig)
                      rope_freq_base_, // 8.  Base frequency theta (e.g.,
                                       // 1,000,000.0f for Qwen)
                      1.0f,            // 9.  Frequency scale factor
                      0.0f, // 10. YaRN extrapolation factor (ext_factor)
                      1.0f, // 11. Attention scale factor (attn_factor)
                      0.0f, // 12. YaRN low-frequency cutoff (beta_fast)
                      0.0f  // 13. YaRN high-frequency cutoff (beta_slow)
    );
    K = ggml_rope_ext(ctx_, // 1.  Computation context
                      K,    // 2.  Input tensor to rotate (Query or Key)
                      NULL, // 3.  Custom position tensor (optional)
                      0,    // 4.  Sequence position offset (n_past)
                      0,    // 5.  Number of rotated dimensions (n_dims)
                      0,    // 6.  RoPE mode / style flags
                      0,    // 7.  Original context training length (n_ctx_orig)
                      rope_freq_base_, // 8.  Base frequency theta (e.g.,
                                       // 1,000,000.0f for Qwen)
                      1.0f,            // 9.  Frequency scale factor
                      0.0f, // 10. YaRN extrapolation factor (ext_factor)
                      1.0f, // 11. Attention scale factor (attn_factor)
                      0.0f, // 12. YaRN low-frequency cutoff (beta_fast)
                      0.0f  // 13. YaRN high-frequency cutoff (beta_slow)
    );
  }
  int d_head = x->ne[0] / n_heads_;
  ggml_tensor *Q_transformed =
      ggml_reshape_4d(ctx_, Q, d_head, x->ne[1], n_heads_, 1);
  ggml_tensor *K_transformed =
      ggml_reshape_4d(ctx_, K, d_head, x->ne[1], n_kv_, 1);
  ggml_tensor *V_transformed =
      ggml_reshape_4d(ctx_, V, d_head, x->ne[1], n_kv_, 1);
  ggml_tensor *KQ = ggml_mul_mat(
      ctx_, ggml_repeat(ctx_, K_transformed, Q_transformed), Q_transformed);
  const float scale = 1.0f / sqrtf((float)Q->ne[0]);
  KQ = ggml_scale(ctx_, KQ, scale);
  KQ = ggml_diag_mask_inf(ctx_, KQ, x->ne[1]);
  ggml_tensor *scaled_attn =
      ggml_mul_mat(ctx_, V_transformed, ggml_soft_max(ctx_, KQ));
  ggml_tensor *attn_out = ggml_mul_mat(
      ctx_, ggml_reshape_2d(ctx_, scaled_attn, x->ne[0], x->ne[1]), out_w_);
  return attn_out;
}
