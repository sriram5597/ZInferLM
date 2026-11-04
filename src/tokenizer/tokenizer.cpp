#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

#include "tokenizer_impl.h"

namespace zinferlm {

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;

Tokenizer::Tokenizer(Tokenizer &&) noexcept = default;
Tokenizer &Tokenizer::operator=(Tokenizer &&) noexcept = default;

Tokenizer Tokenizer::for_model(zinferlm::Model &model) {
  Tokenizer t;
  t.impl_ = internal::TokenizerImpl::for_model(model);
  return t;
}

void Tokenizer::init() {}

std::vector<std::string> Tokenizer::pretokenize(std::string text) {
  return impl_->pretokenize(std::move(text));
}

std::vector<int32_t> Tokenizer::tokenize(std::string text) {
  return impl_->tokenize(std::move(text));
}

std::string Tokenizer::decode(uint32_t token_id) const {
  return impl_->decode(token_id);
}

bool Tokenizer::is_stop_token(std::string token) {
  return impl_->is_stop_token(std::move(token));
}

std::string Tokenizer::get_bos() { return impl_->get_bos(); }

std::string Tokenizer::get_eos() { return impl_->get_eos(); }

} // namespace zinferlm
