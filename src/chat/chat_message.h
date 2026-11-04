#pragma once

#include <string>
#include <vector>

#include <minja/chat-template.hpp>
#include <zinferlm/chat.h>

namespace zinferlm {
namespace internal {

class ChatTemplate {
private:
  minja::chat_template tmpl_;
  std::string template_str_;
  ChatTemplate(std::string t, std::string bos_token, std::string eos_token)
      : tmpl_(t, bos_token, eos_token) {}

public:
  static ChatTemplate from_str(std::string template_str, std::string bos_token,
                               std::string eos_token);
  std::string render(std::vector<zinferlm::ChatMessage> messages,
                     bool add_generation_prompt);
};

} // namespace internal
} // namespace zinferlm
