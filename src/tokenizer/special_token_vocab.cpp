#include <cstdint>
#include <zinferlm/tokenizer.h>
#include <memory>
#include <string>
#include <unordered_map>

SpecialTokenTrie::SpecialTokenTrie() {
  root = std::make_unique<trie_node_t>('\0');
}

void SpecialTokenTrie::build_from_tokens(std::vector<std::string> &tokens,
                                         std::vector<int32_t> &token_types) {
  int s= 0;
  root = std::make_unique<trie_node_t>('\0');
  for (int i = 0; i < tokens.size(); i++) {
    if (token_types[i] == zinferlm::token_type::TOKEN_TYPE_NORMAL) {
      continue;
    }
    std::string tok = tokens[i];
    trie_node_t *current = root.get();
    for (int j = 0; j < tok.size(); j++) {
      auto &child = current->next[tok[j]];
      if (!child) {
        child = std::make_unique<trie_node_t>(tok[j]);
      }
      current = child.get();
    }
    current->token_id = i;
    s++;
  }
  size = s;
}

int SpecialTokenTrie::match(std::string &text, int start, int end) {
  trie_node_t *current = root.get();
  int i = start;
  int token_id = -1;
  while (i < end && current != nullptr) {
    auto &next = current->next[text[i]];
    if (!next) {
      break;
    }
    current = next.get();
    if (current->token_id >= 0) {
      token_id = current -> token_id;
    }
    i++;
  }
  return token_id;
}

