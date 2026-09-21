#include "zinferlm/models.h"
#include <cstdint>
#include <string>
#include <ggml.h>
#include <vector>

ggml_tensor* create_tensor(ggml_context* ctx, zinferlm::tensor_info_t info);
