#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <ggml.h>
#include <zinferlm/models.h>

namespace zinferlm {
namespace internal {

ggml_tensor *create_tensor(ggml_context *ctx, zinferlm::tensor_info_t info);

} // namespace internal
} // namespace zinferlm
