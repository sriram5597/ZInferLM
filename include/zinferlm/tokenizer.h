#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

#include "models.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

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

namespace zinferlm {
enum token_type {
  TOKEN_TYPE_UNDEFINED = 0,
  TOKEN_TYPE_NORMAL,
  TOKEN_TYPE_UNKNOWN,
  TOKEN_TYPE_CONTROL,
  TOKEN_TYPE_USER_DEFINED,
  TOKEN_TYPE_UNUSED,
  TOKEN_TYPE_BYTE
};

class Tokenizer {
private:
  using Pcre2CodePtr = std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)>;
  using Pcre2DataPtr =
      std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)>;

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

  Tokenizer(std::string m, std::string p, std::vector<std::string> t,
            std::vector<int32_t> token_types)
      : model_(m), pre_(p), tokens_(t), token_types_(token_types) {
    spl_token_trie_ = SpecialTokenTrie();
    stop_tokens_ = {"<|im_end|>", "<|endoftext|>",

                    // LLaMA 3 / 3.1 / 3.2
                    "<|eot_id|>", "<|end_of_text|>",

                    // LLaMA 1/2, Mistral, SentencePiece base
                    "</s>", "[/INST]",

                    // Gemma
                    "<end_of_turn>", "<eos>",

                    // DeepSeek
                    "<｜end of sentence｜>",

                    // Phi / Command-R
                    "<|end|>", "<|END_OF_TURN_TOKEN|>"};
  }
  void init_pretokenize_();
  void build_byte_to_unicode_map_();
  void build_token_id_map_();
  void build_merges_map_(std::vector<std::string> merges);
  std::vector<std::string> split_and_merge_(std::string chunk);
  std::string unicode_to_chars_(std::string s) const;
  void bpe_(std::string text, std::vector<std::string> &tokens);

public:
  static Tokenizer for_model(zinferlm::Model &model);
  void init();
  std::vector<std::string> pretokenize(std::string text);
  std::vector<int32_t> tokenize(std::string text);
  std::string decode(uint32_t token_id) const;
  bool is_stop_token(std::string token);
};
}; // namespace zinferlm
