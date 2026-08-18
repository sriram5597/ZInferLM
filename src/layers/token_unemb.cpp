#include <ggml-cpp.h>
#include <ggml.h>

#include <zinferlm/models.h>
#include "layers.h"

TokenUnembedding::TokenUnembedding(token_unembedding_params_t params) : ctx_(params.ctx), unemb_w_(params.unemb_w) {}

ggml_tensor *TokenUnembedding::operator()(ggml_tensor *x) const
{
    return ggml_mul_mat(ctx_, unemb_w_, x);
}
