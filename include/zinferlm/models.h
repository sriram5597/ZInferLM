#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

struct ggml_context;
struct ggml_backend;
class Layer;
class KVCache;
class Graph;

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
  u_int32_t eos_token_id;
  u_int32_t padding_token_id;
  u_int32_t bos_token_id;
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
  void *data;
  std::vector<uint64_t> dimensions;
};

enum StreamStatus {
  CONTINUE,
  EOS,
  MAX_CTX_REACHED
};

class Model {
public:
  virtual ~Model() = default;
  virtual model_info_t info() const = 0;
  virtual tokenizer_info_t tokenizer_info() const = 0;
  virtual std::vector<tensor_info_t> tensor_info() const = 0;
  virtual model_config_t config() const = 0;
  virtual std::vector<std::unique_ptr<Layer>> create_layers(ggml_context *ctx,
                                                            int past_tokens,
                                                            int len,
                                                            KVCache *cache) = 0;
  virtual void summary() = 0;

  std::vector<float> predict(std::vector<int32_t> tokens, int past_tokens);

  static Model &instance();
  static bool load(const char *model_path);

  void set_debug(bool enabled) { debug_ = enabled; }
  std::string invoke(std::string input, int max_tokens);

protected:
  Model() = default;
  bool debug_ = false;
  std::unique_ptr<KVCache> cache_;
  std::unique_ptr<ggml_backend, void (*)(ggml_backend *)> backend_{
      nullptr, [](ggml_backend *) {}};

private:
  static std::unique_ptr<Model> instance_;
};
} // namespace zinferlm
