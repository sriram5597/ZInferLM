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
    u_int64_t eos_token_id;
    u_int64_t padding_token_id;
    u_int64_t bos_token_id;
    bool add_bos_token;
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<int> token_type;
  };

  class Model
  {
  public:
    virtual ~Model() = default;
    virtual model_info_t info() const = 0;
    virtual tokenizer_info_t tokenizer_info() const = 0;

    static Model &instance();
    static bool load(const char *model_path);

  protected:
    Model() = default;

  private:
    static std::unique_ptr<Model> instance_;
  };
}
