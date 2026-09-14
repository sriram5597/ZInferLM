#include "ggml.h"
#include "layers.h"

SwigLU::SwigLU(swiglu_params_t params)
    : Layer(params.ctx, params.residual, params.norm_gamma, params.norm_eps),
      w_gate_(params.w_gate), w_up_(params.w_up), w_down_(params.w_down) {}

ggml_tensor *SwigLU::forward(ggml_tensor *x) const {
  ggml_tensor *gate = ggml_mul_mat(ctx_, w_gate_, x);
  ggml_tensor *up = ggml_mul_mat(ctx_, w_up_, x);

  gate = ggml_silu(ctx_, gate);
  ggml_tensor *swiglu = ggml_mul(ctx_, gate, up);
  return ggml_mul_mat(ctx_, w_down_, swiglu);
}
