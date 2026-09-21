#include <cstdint>
#include <ggml.h>
#include <iostream>
#include <string>
#include <vector>

#include "tensors.h"
#include "zinferlm/models.h"

ggml_tensor *create_tensor(ggml_context *ctx, zinferlm::tensor_info_t info) {
  ggml_tensor *t =
      ggml_new_tensor(ctx, static_cast<ggml_type>(info.type_id), info.n_dim,
                      reinterpret_cast<int64_t *>(info.dimensions.data()));
  t->data = info.data;
  ggml_set_name(t, info.name.c_str());
  return t;
}
