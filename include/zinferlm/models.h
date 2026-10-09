#pragma once

#include <cstdint>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class KVCache;

using Backend = enum ggml_backend_dev_type;

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
  std::vector<uint64_t> dimensions;
};

class ModelLoader {
public:
  virtual zinferlm::model_info_t info() const = 0;
  virtual zinferlm::tokenizer_info_t tokenizer_info() const = 0;
  virtual std::vector<zinferlm::tensor_info_t> tensor_info() const = 0;
  virtual void *get_tensor_ptr(uint64_t offset) const = 0;
  virtual uint64_t get_tensor_count() const = 0;
  virtual zinferlm::tensor_info_t tensor_info(std::string id) const = 0;
  virtual zinferlm::model_config_t model_config() const = 0;
};

enum StreamStatus { CONTINUE, EOS, MAX_CTX_REACHED };

class Model {
public:
  model_info_t info() const;
  tokenizer_info_t tokenizer_info() const;
  std::vector<tensor_info_t> tensor_info() const;
  model_config_t config() const;

  virtual ggml_tensor *build_graph(ggml_context *ctx, ggml_cgraph *gf,
                                   KVCache *cache) = 0;
  virtual void create_input_tensors(ggml_context *ctx, uint32_t n_tokens,
                                    uint32_t max_len) = 0;
  virtual std::vector<ggml_tensor *> get_input_tensors() = 0;
  virtual void set_inputs(const std::vector<int32_t> &tokens, int past_tokens) = 0;
  virtual void summary() = 0;

  ggml_backend *get_backend();
  std::vector<ggml_backend *> backends() const;
  static Backend get_backend_type();

  static Model &instance();
  static bool load(const char *model_path);

protected:
  ggml_context_ptr tensor_ctx_;
  std::unordered_map<std::string, ggml_tensor *> tensor_map_;
  std::unique_ptr<ModelLoader> loader_;
  std::vector<ggml_backend_ptr> backends_;
  Model(zinferlm::ModelLoader *loader);

  void load_tensors_();
  ggml_tensor *get_tensor_(std::string);

private:
  ggml_backend_buffer_ptr tensor_buffer_;
  static std::unique_ptr<Model> instance_;
};
} // namespace zinferlm
