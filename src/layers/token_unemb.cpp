#include <ggml-cpp.h>
#include <ggml.h>

#include "layers.h"

namespace zinferlm {
namespace internal {

TokenUnembedding::TokenUnembedding(token_unembedding_params_t params)
    : Layer(params.ctx, false, params.norm_gamma, params.norm_eps),
      unemb_w_(params.unemb_w) {
  name = "TokenUnembedding";
}

ggml_tensor *TokenUnembedding::forward(ggml_tensor *x) const {
  return ggml_mul_mat(ctx_, unemb_w_, x);
}

} // namespace internal
} // namespace zinferlm
