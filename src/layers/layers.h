#pragma once

#include <cstdint>
#include <ggml-cpp.h>
#include <ggml.h>

#include <zinferlm/models.h>

class Layer {
public:
  std::string name;
  virtual ggml_tensor *operator()(ggml_tensor *x) const = 0;
};

struct token_embedding_params_t {
  ggml_context *ctx;
  ggml_tensor *emb_w;
};

class TokenEmbedding : public Layer {
private:
  ggml_tensor *emb_w_;
  ggml_context *ctx_;

public:
  TokenEmbedding(token_embedding_params_t params);
  ggml_tensor *operator()(ggml_tensor *x) const override;
};

struct token_unembedding_params_t {
  ggml_context *ctx;
  ggml_tensor *unemb_w;
};

class TokenUnembedding : public Layer {
private:
  ggml_tensor *unemb_w_;
  ggml_context *ctx_;

public:
  TokenUnembedding(token_unembedding_params_t params);
  ggml_tensor *operator()(ggml_tensor *x) const override;
};

struct grouped_attn_head_params {
  ggml_context *ctx;
  ggml_tensor *q_w, *k_w, *q_b, *k_b, *v_w, *v_b, *out_w, *out_b;
  float rope_freq_base;
  bool apply_rope;
  uint32_t n_heads;
  uint32_t n_kv;
  uint64_t d_model_;
};

class GroupedAttentionHead : public Layer {
private:
  ggml_tensor *q_w_, *q_b_, *k_w_, *k_b_, *v_w_, *v_b_, *out_w_, *out_b_;
  ggml_context *ctx_;
  bool apply_rope_;
  float rope_freq_base_;
  uint32_t n_heads_, n_kv_;

public:
  GroupedAttentionHead(grouped_attn_head_params);
  ggml_tensor *operator()(ggml_tensor *x) const override;
};

struct swiglu_params_t {
  ggml_context *ctx;
  ggml_tensor *w_gate;
  ggml_tensor *w_down;
  ggml_tensor *w_up;
};

class SwigLU : public Layer {
private:
  ggml_context *ctx_;
  ggml_tensor *w_gate_, *w_up_, *w_down_;

public:
  SwigLU(swiglu_params_t params);
  ggml_tensor *operator()(ggml_tensor *x) const override;
};

struct norm_params_t {
  ggml_context *ctx;
  ggml_tensor *gamma;
  float eps;
};

class NormLayer : public Layer {
private:
  ggml_context *ctx_;
  ggml_tensor *gamma_;
  float eps_;

public:
  NormLayer(norm_params_t params);
  ggml_tensor *operator()(ggml_tensor *x) const override;
};
