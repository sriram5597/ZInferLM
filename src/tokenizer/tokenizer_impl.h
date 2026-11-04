#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

namespace zinferlm {
namespace internal {

enum token_type {
  TOKEN_TYPE_UNDEFINED = 0,
  TOKEN_TYPE_NORMAL,
  TOKEN_TYPE_UNKNOWN,
  TOKEN_TYPE_CONTROL,
  TOKEN_TYPE_USER_DEFINED,
  TOKEN_TYPE_UNUSED,
  TOKEN_TYPE_BYTE
};

struct trie_node_t {
  char ch;
  std::unordered_map<char, std::unique_ptr<trie_node_t>> next;
  int token_id = -1;

  explicit trie_node_t(char c) : ch(c) {}
};

class SpecialTokenTrie {
private:
  std::unique_ptr<trie_node_t> root;

public:
  uint32_t size;
  SpecialTokenTrie();
  void build_from_tokens(std::vector<std::string> &tokens,
                         std::vector<int32_t> &token_types);
  int match(std::string &text, int start, int end);
};

inline const std::string PRE_TOKENIZER_QWEN2_REGEX =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
inline const std::string PRE_TOKENIZER_LLAMA_REGEX =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|[^"
    "\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

class TokenizerImpl {
private:
  using Pcre2CodePtr = std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)>;
  using Pcre2DataPtr =
      std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)>;

  tokenizer_info_t info_;
  std::string model_;
  std::string pre_;
  std::unordered_map<std::string, uint64_t> merges_map_;
  std::vector<int32_t> token_types_;
  Pcre2CodePtr re_{nullptr, pcre2_code_free};
  Pcre2DataPtr match_data_{nullptr, pcre2_match_data_free};
  std::unordered_map<uint8_t, std::string> byte_to_unicode_map_;
  std::unordered_map<std::string, uint8_t> unicode_to_byte_map_;
  std::vector<std::string> tokens_;
  std::unordered_map<std::string, uint64_t> token_to_id_map_;
  std::unordered_map<uint64_t, std::string> id_to_token_map_;
  std::set<std::string> stop_tokens_;
  SpecialTokenTrie spl_token_trie_;

  void init_pretokenize_();
  void build_byte_to_unicode_map_();
  void build_token_id_map_();
  void build_merges_map_(std::vector<std::string> merges);
  std::vector<std::string> split_and_merge_(std::string chunk);
  std::string unicode_to_chars_(std::string s) const;
  void bpe_(std::string text, std::vector<std::string> &tokens);

public:
  TokenizerImpl(std::string m, std::string p, std::vector<std::string> t,
                std::vector<int32_t> token_types);

  static std::unique_ptr<TokenizerImpl> for_model(zinferlm::Model &model);

  std::vector<std::string> pretokenize(std::string text);
  std::vector<int32_t> tokenize(std::string text);
  std::string decode(uint32_t token_id) const;
  bool is_stop_token(std::string token);
  std::string get_bos();
  std::string get_eos();
};

} // namespace internal
} // namespace zinferlm
