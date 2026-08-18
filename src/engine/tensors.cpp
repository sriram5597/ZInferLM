#include <cstdint>
#include <ggml.h>
#include <vector>

#include "tensors.h"

ggml_tensor* create_tensor(ggml_context* ctx, ggml_type type, void* data, std::vector<uint64_t> dims) {
  ggml_tensor* t = ggml_new_tensor(ctx, type, dims.size(), reinterpret_cast<int64_t *>(dims.data()));
  t->data = data;
  return t;
}
