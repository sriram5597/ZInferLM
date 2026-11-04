#include "layers.h"

namespace zinferlm {
namespace internal {

Block::Block(block_params_t params) : Layer(params.ctx), layers_(params.layers) {
  name = "Block";
}

ggml_tensor *Block::forward(ggml_tensor *x) const {
  ggml_tensor *out = x;
  for (auto &&l : layers_) {
    out = (*l)(out);
  }
  return out;
}

} // namespace internal
} // namespace zinferlm
