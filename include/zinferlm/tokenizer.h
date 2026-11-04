#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "zinferlm/export.h"
#include "zinferlm/models.h"

namespace zinferlm {

namespace internal {
class TokenizerImpl;
}

class ZINFERLM_API Tokenizer {
public:
  Tokenizer();
  ~Tokenizer();

  Tokenizer(const Tokenizer &) = delete;
  Tokenizer &operator=(const Tokenizer &) = delete;
  Tokenizer(Tokenizer &&) noexcept;
  Tokenizer &operator=(Tokenizer &&) noexcept;

  static Tokenizer for_model(zinferlm::Model &model);

  void init();
  std::vector<std::string> pretokenize(std::string text);
  std::vector<int32_t> tokenize(std::string text);
  std::string decode(uint32_t token_id) const;
  bool is_stop_token(std::string token);
  std::string get_bos();
  std::string get_eos();

private:
  std::unique_ptr<internal::TokenizerImpl> impl_;
};

} // namespace zinferlm
