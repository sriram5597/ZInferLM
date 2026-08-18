#include <cstdint>
#include <ggml.h>
#include <vector>

ggml_tensor* create_tensor(ggml_context* ctx, ggml_type type,  void* data, std::vector<uint64_t> dims);
