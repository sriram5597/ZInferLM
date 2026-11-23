#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <zinferlm/config.hpp>

namespace zinferlm {
namespace internal {

// One line of the memory budget.
struct memory_block_t {
  std::string name;
  std::string backend; // empty for host allocations
  std::string formula;
  size_t bytes = 0;
};

// Full memory plan for a model under a given runtime configuration.
//
// Produced without allocating weights, KV cache or compute buffers: the model
// is parsed from its GGUF metadata, tensor structs are created without data,
// and ggml's allocator reports the compute sizes in no-alloc mode.
struct memory_plan_t {
  std::string model_name;
  std::string file_type;
  size_t file_size = 0;
  size_t tensor_count = 0;

  std::vector<memory_block_t> blocks;

  size_t weights = 0;
  size_t tensor_metadata = 0;
  size_t kv_cache = 0;
  size_t compute = 0;
  size_t graph_metadata = 0;
  size_t host_scratch = 0;
  size_t total = 0;

  std::vector<std::string> warnings;
};

// Plans memory for `model_path` with `cfg`. Throws std::runtime_error when the
// file cannot be read or is not a GGUF file.
memory_plan_t plan_memory(const char *model_path, const config_t &cfg,
                          bool cpu_only = false);

} // namespace internal
} // namespace zinferlm
