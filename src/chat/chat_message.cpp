#include <minja/minja.hpp>
#include <nlohmann/json.hpp>
#include <string.h>

#include "chat_message.h"
#include "minja/chat-template.hpp"

ChatTemplate ChatTemplate::from_str(std::string template_str,
                                    std::string bos_token,
                                    std::string eos_token) {
  ChatTemplate t(template_str, bos_token, eos_token);
  return t;
}

std::string ChatTemplate::render(std::vector<zinferlm::ChatMessage> messages,
                                 bool add_generation_prompt) {
  json json_messages = json::array();
  for (const auto m : messages) {
    json_messages.push_back({{"role", m.role}, {"content", m.message}});
  }
  json context = {{"messages", json_messages},
                  {"add_generation_prompt", add_generation_prompt}};
  try {
    minja::chat_template_inputs inputs;
    inputs.messages = json_messages;
    inputs.add_generation_prompt = add_generation_prompt;
    return tmpl_.apply(inputs);
  } catch (const std::exception err) {
    std::cout << "Failed to render template: " << err.what() << std::endl;
    return "";
  }
}
