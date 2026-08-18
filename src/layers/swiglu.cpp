#include "ggml.h"
#include "layers.h"

SwigLU::SwigLU(swiglu_params_t params)
    : w_gate_(params.w_gate), w_up_(params.w_up), w_down_(params.w_down),
      ctx_(params.ctx) {}

ggml_tensor *SwigLU::operator()(ggml_tensor *x) const {
  ggml_tensor *gate = ggml_mul_mat(ctx_, w_gate_, x);
  ggml_tensor *up = ggml_mul_mat(ctx_, w_up_, x);

  gate = ggml_silu(ctx_, gate);
  ggml_tensor *swiglu = ggml_mul_mat(ctx_, gate, up);

  return ggml_mul_mat(ctx_, w_down_, swiglu);
}
