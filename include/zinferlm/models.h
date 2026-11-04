#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "zinferlm/export.h"

namespace zinferlm {

struct model_info_t {
  std::string name;
  std::string version;
  std::string architecture;
  std::string file_type;
};

struct model_config_t {
  uint32_t nheads;
  uint32_t nkv;
  float rope_freq_base;
  uint32_t n_blocks;
  uint64_t embedding_dim;
  float rms_eps;
  uint32_t max_context_len;
};

struct tokenizer_info_t {
  std::string model;
  std::string pre;
  uint32_t eos_token_id;
  uint32_t padding_token_id;
  uint32_t bos_token_id;
  bool add_bos_token;
  std::vector<std::string> tokens;
  std::vector<std::string> merges;
  std::vector<int32_t> token_type;
  std::string template_str;
};

struct tensor_info_t {
  std::string name;
  std::string type_name;
  uint32_t type_id;
  uint32_t n_dim;
  uint64_t data_offset;
  std::vector<uint64_t> dimensions;
};

enum StreamStatus { CONTINUE, EOS, MAX_CTX_REACHED };

// Opaque implementation details. Declared here only so the facade can hold a
// pointer to them; their definitions live in src/ and are never exposed.
namespace internal {
class ModelImpl;
}

// Public model facade. Hides all ggml/backend/graph internals behind an
// opaque ModelImpl. Use load()/instance()/unload() to manage the singleton.
class ZINFERLM_API Model {
public:
  Model();
  ~Model();

  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;
  Model(Model &&) = delete;
  Model &operator=(Model &&) = delete;

  static bool load(const char *model_path);
  static void unload();
  static Model &instance();

  model_info_t info() const;
  tokenizer_info_t tokenizer_info() const;
  std::vector<tensor_info_t> tensor_info() const;
  model_config_t config() const;
  void summary();

  // Opaque accessor for library-internal callers only. The returned type is
  // incomplete here; external code cannot do anything with it.
  internal::ModelImpl &impl();

private:
  std::unique_ptr<internal::ModelImpl> impl_;
  static std::unique_ptr<Model> instance_;
};

} // namespace zinferlm
