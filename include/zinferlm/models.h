#pragma once

#include <string>
#include <memory>
#include <vector>

namespace zinferlm
{
  struct model_info_t
  {
    std::string name;
    std::string version;
    std::string architecture;
    std::string file_type;
  };

  struct tokenizer_info_t
  {
    std::string model;
    std::string pre;
    u_int32_t eos_token_id;
    u_int32_t padding_token_id;
    u_int32_t bos_token_id;
    bool add_bos_token;
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<int32_t> token_type;
  };

  struct tensor_info_t
  {
    std::string name;
    std::string type_name;
    uint32_t type_id;
    uint32_t n_dim;
    uint64_t data_offset;
    std::vector<uint64_t> dimensions;
  };

  class Model
  {
  public:
    virtual ~Model() = default;
    virtual model_info_t info() const = 0;
    virtual tokenizer_info_t tokenizer_info() const = 0;
    virtual std::vector<tensor_info_t> tensor_info() const = 0;
    virtual std::string invoke(std::string input) = 0;
    virtual std::vector<uint32_t> tokenize(std::string input) = 0;
    virtual void summary(std::string input) = 0;

    static Model &instance();
    static bool load(const char *model_path);

  protected:
    Model() = default;

  private:
    static std::unique_ptr<Model> instance_;
  };
}
