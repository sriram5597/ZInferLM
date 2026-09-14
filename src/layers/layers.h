#pragma once

#include <cstdint>
#include <ggml-cpp.h>
#include <ggml.h>

#include <vector>
#include <zinferlm/models.h>

enum Layers {
  GROUPED_ATTN,
  TOKEN_EMBEDDING,
  TOKEN_UNEMBEDDING,
  SWIGLU,
  BLOCK,
};

class Layer {
protected:
  ggml_context *ctx_;
  bool residual_ = false;
  ggml_tensor *pre_norm_gamma_ = nullptr;
  float pre_norm_eps_ = 0.0f;
  virtual ggml_tensor *forward(ggml_tensor *x) const = 0;

public:
  Layer(ggml_context *ctx) : ctx_(ctx) {}
  Layer(ggml_context *ctx, bool residual, ggml_tensor *pre_norm_gamma,
        float pre_norm_eps)
      : ctx_(ctx), residual_(residual), pre_norm_gamma_(pre_norm_gamma),
        pre_norm_eps_(pre_norm_eps) {}
  virtual ~Layer() = default;
  std::string name;
  ggml_tensor *operator()(ggml_tensor *x) const {
    ggml_tensor *in = x;
    if (pre_norm_gamma_ != nullptr) {
      in = ggml_mul(ctx_, ggml_rms_norm(ctx_, x, pre_norm_eps_),
                    pre_norm_gamma_);
    }
    ggml_tensor *out = forward(in);
    return residual_ ? ggml_add(ctx_, out, x) : out;
  }
  void set_residual(bool v) { residual_ = v; }
};

struct token_embedding_params_t {
  ggml_context *ctx;
  ggml_tensor *emb_w;
};

class TokenEmbedding : public Layer {
private:
  ggml_tensor *emb_w_;

public:
  TokenEmbedding(token_embedding_params_t params);
  ggml_tensor *forward(ggml_tensor *x) const override;
};

struct token_unembedding_params_t {
  ggml_context *ctx;
  ggml_tensor *unemb_w;
  ggml_tensor *norm_gamma;
  float norm_eps;
};

class TokenUnembedding : public Layer {
private:
  ggml_tensor *unemb_w_;

public:
  TokenUnembedding(token_unembedding_params_t params);
  ggml_tensor *forward(ggml_tensor *x) const override;
};

struct grouped_attn_head_params {
  ggml_context *ctx;
  ggml_tensor *q_w, *k_w, *q_b, *k_b, *v_w, *v_b, *out_w, *out_b;
  float rope_freq_base;
  bool apply_rope;
  uint32_t n_heads;
  uint32_t n_kv;
  uint64_t d_model;
  int past_tokens;
  int len;
  bool residual;
  ggml_tensor *norm_gamma;
  float norm_eps;
};

class GroupedAttentionHead : public Layer {
private:
  ggml_tensor *q_w_, *q_b_, *k_w_, *k_b_, *v_w_, *v_b_, *out_w_, *out_b_;
  bool apply_rope_;
  float rope_freq_base_;
  uint32_t n_heads_, n_kv_;
  int past_tokens_, len_;

public:
  GroupedAttentionHead(grouped_attn_head_params);
  ggml_tensor *forward(ggml_tensor *x) const override;
};

struct swiglu_params_t {
  ggml_context *ctx;
  ggml_tensor *w_gate;
  ggml_tensor *w_down;
  ggml_tensor *w_up;
  bool residual;
  ggml_tensor *norm_gamma;
  float norm_eps;
};

class SwigLU : public Layer {
private:
  ggml_tensor *w_gate_, *w_up_, *w_down_;

public:
  SwigLU(swiglu_params_t params);
  ggml_tensor *forward(ggml_tensor *x) const override;
};

struct block_params_t {
  ggml_context *ctx;
  std::vector<Layer *> layers;
};

class Block : public Layer {
private:
  std::vector<Layer *> layers_;

public:
  Block(block_params_t params);
  ggml_tensor *forward(ggml_tensor *x) const override;
};
