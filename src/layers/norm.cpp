#include <iostream>
#include "ggml.h"
#include "layers.h"

NormLayer::NormLayer(norm_params_t params)
    : Layer(params.ctx), gamma_(params.gamma), eps_(params.eps) {
      name = "NormLayer";
    }

ggml_tensor *NormLayer::forward(ggml_tensor *x) const {
  ggml_tensor *norm = ggml_rms_norm(ctx_, x, eps_);
  return ggml_mul(ctx_, norm, gamma_);
}
