#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "tokenizer.h"
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

std::string get_pretokenizer_regex(std::string model) {
  if (model == "qwen2") {
    return PRE_TOKENIZER_QWEN2_REGEX;
  }
  if (model == "llama-bpe") {
    return PRE_TOKENIZER_LLAMA_REGEX;
  }
  return "";
}

void zinferlm::Tokenizer::build_byte_to_unicode_map_() {
  auto code_point_to_utf8 = [](uint32_t cp) -> std::string {
    std::string out;
    if (cp <= 0x7F) {
      out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
  };

  std::vector<bool> is_printable(256, false);

  // Mark printable ranges (ASCII '!' to '~', Latin-1 '¡' to '¬', '®' to 'ÿ')
  for (int b = '!'; b <= '~'; ++b)
    is_printable[b] = true;
  for (int b = 0xA1; b <= 0xAC; ++b)
    is_printable[b] = true;
  for (int b = 0xAE; b <= 0xFF; ++b)
    is_printable[b] = true;

  uint32_t shifted_code_point =
      256; // Start shifting unprintable bytes above U+0100

  for (int b = 0; b < 256; ++b) {
    if (is_printable[b]) {
      std::string unicode = code_point_to_utf8(b);
      byte_to_unicode_map_[static_cast<uint8_t>(b)] = unicode;
      unicode_to_byte_map_[unicode] = static_cast<uint8_t>(b);
    } else {
      // Shift unprintable bytes into U+0100 range
      std::string unicode = code_point_to_utf8(shifted_code_point);
      byte_to_unicode_map_[static_cast<uint8_t>(b)] = unicode;
      unicode_to_byte_map_[unicode] = static_cast<uint8_t>(b);
      shifted_code_point++;
    }
  }
}

void zinferlm::Tokenizer::build_merges_map_(std::vector<std::string> merges) {
  for (int i = 0; i < merges.size(); i++) {
    merges_map_[merges[i]] = i;
  }
}

void zinferlm::Tokenizer::build_token_id_map_() {
  for (int i = 0; i < tokens_.size(); i++) {
    token_to_id_map_[tokens_[i]] = i;
    id_to_token_map_[i] = tokens_[i];
  }
}

void zinferlm::Tokenizer::init_pretokenize_() {
  int ec;
  PCRE2_SIZE err_offset;

  std::string regex = get_pretokenizer_regex(pre_);
  pcre2_code *compiled_re = pcre2_compile(
      reinterpret_cast<PCRE2_SPTR>(regex.c_str()), PCRE2_ZERO_TERMINATED,
      PCRE2_UTF | PCRE2_UCP, &ec, &err_offset, nullptr);
  if (!compiled_re) {
    PCRE2_UCHAR buffer[256];
    pcre2_get_error_message(ec, buffer, sizeof(buffer));
    throw std::runtime_error("PCRE2 Compilation Failed at offset " +
                             std::to_string(err_offset) + ": " +
                             reinterpret_cast<char *>(buffer));
  }
  re_.reset(compiled_re);
  pcre2_jit_compile(re_.get(), PCRE2_JIT_COMPLETE);
  match_data_.reset(pcre2_match_data_create_from_pattern(re_.get(), nullptr));
}

zinferlm::Tokenizer zinferlm::Tokenizer::for_model(zinferlm::Model &model) {
  tokenizer_info_t info = model.tokenizer_info();
  zinferlm::Tokenizer obj =
      Tokenizer(info.model, info.pre, info.tokens, info.token_type);
  obj.init_pretokenize_();
  obj.build_byte_to_unicode_map_();
  obj.build_token_id_map_();
  obj.build_merges_map_(info.merges);
  obj.spl_token_trie_.build_from_tokens(info.tokens, info.token_type);
  return obj;
}

std::vector<std::string> zinferlm::Tokenizer::pretokenize(std::string text) {
  std::string pre_token_regex = get_pretokenizer_regex(pre_);
  std::vector<std::string> chunks;

  PCRE2_SPTR text_to_split = reinterpret_cast<PCRE2_SPTR>(text.c_str());
  PCRE2_SIZE tex_len = text.size();
  PCRE2_SIZE start_offset = 0;

  while (start_offset < tex_len) {
    int rc = pcre2_jit_match(re_.get(), text_to_split, tex_len, start_offset, 0,
                             match_data_.get(), nullptr);

    if (rc < 0) {
      if (rc == PCRE2_ERROR_NOMATCH) {
        chunks.push_back(text.substr(start_offset, 1));
        start_offset++;
        continue;
      }
      throw std::runtime_error("PCRE2 matching error code: " +
                               std::to_string(rc));
    }
    PCRE2_SIZE *offset_vector = pcre2_get_ovector_pointer(match_data_.get());
    PCRE2_SIZE match_start = offset_vector[0];
    PCRE2_SIZE match_end = offset_vector[1];

    chunks.push_back(text.substr(match_start, match_end - match_start));
    start_offset = match_end;
  }
  return chunks;
}

std::vector<std::string>
zinferlm::Tokenizer::split_and_merge_(std::string chunk) {
  std::vector<std::string> tokens;
  std::vector<std::pair<int, uint64_t>> rank_pairs;
  for (char c : chunk) {
    tokens.push_back(byte_to_unicode_map_[static_cast<uint8_t>(c)]);
  }
  while (true) {
    rank_pairs.clear();
    for (int i = 0; i < tokens.size() - 1; i++) {
      auto rank = merges_map_.find(tokens[i] + " " + tokens[i + 1]);
      if (rank != merges_map_.end()) {
        rank_pairs.push_back(std::make_pair(i, rank->second));
      }
    }
    if (rank_pairs.size() == 0) {
      break;
    }
    auto it = std::min_element(
        rank_pairs.begin(), rank_pairs.end(),
        [](const auto &a, const auto &b) { return a.second < b.second; });
    if (it != rank_pairs.end()) {
      int merge_index = it->first;
      tokens[merge_index] += tokens[merge_index + 1];
      tokens.erase(tokens.begin() + merge_index + 1);
    }
  }
  return tokens;
}

std::string zinferlm::Tokenizer::unicode_to_chars_(std::string token) const {
  std::string out;
  size_t i = 0;
  while (i < token.size()) {
    unsigned char c = static_cast<unsigned char>(token[i]);
    size_t len = 1;
    if ((c & 0xE0) == 0xC0)
      len = 2;
    else if ((c & 0xF0) == 0xE0)
      len = 3;
    else if ((c & 0xF8) == 0xF0)
      len = 4;
    std::string piece = token.substr(i, len);
    auto m = unicode_to_byte_map_.find(piece);
    if (m != unicode_to_byte_map_.end()) {
      out.push_back(static_cast<char>(m->second));
    } else {
      out += piece;
    }
    i += len;
  }
  return out;
}

void zinferlm::Tokenizer::bpe_(std::string text,
                               std::vector<std::string> &tokens) {
  std::vector<std::string> chunks = pretokenize(text);
  for (auto c : chunks) {
    std::vector<std::string> splitted_tokens = split_and_merge_(c);
    tokens.insert(tokens.end(), splitted_tokens.begin(), splitted_tokens.end());
  }
}

std::vector<int32_t> zinferlm::Tokenizer::tokenize(std::string text) {
  std::vector<std::string> final_tokens;
  std::vector<int32_t> token_list;
  int start = 0;
  std::string pretext = "";
  while (start < text.size()) {
    int token_id = spl_token_trie_.match(text, start, text.size());
    if (token_id >= 0) {
      std::string spl_token = id_to_token_map_[token_id];
      bpe_(pretext, final_tokens);
      pretext = "";
      final_tokens.push_back(spl_token);
      start += spl_token.size();
      continue;
    }
    pretext += text[start];
    start++;
  }
  if (pretext.size() > 0) {
    bpe_(pretext, final_tokens);
  }
  for (auto t : final_tokens) {
    token_list.push_back(token_to_id_map_[t]);
  }
  return token_list;
}

std::string zinferlm::Tokenizer::decode(uint32_t token_id) const {
  auto it = id_to_token_map_.find(token_id);
  if (it != id_to_token_map_.end()) {
    return unicode_to_chars_(it->second);
  }
  return "";
}

bool zinferlm::Tokenizer::is_stop_token(std::string token) {
  if (stop_tokens_.contains(token)) {
    if (token_to_id_map_.find(token) != token_to_id_map_.end()) {
      return true;
    }
  }
  return false;
}
