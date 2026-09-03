#include <ggml-cpp.h>
#include <ggml.h>

#include <zinferlm/models.h>
#include "layers.h"

TokenEmbedding::TokenEmbedding(token_embedding_params_t params) : Layer(params.ctx), emb_w_(params.emb_w) {
  name = "TokenEmbedding";
}

ggml_tensor *TokenEmbedding::forward(ggml_tensor *x) const
{
    return ggml_get_rows(ctx_, emb_w_, x);
}
