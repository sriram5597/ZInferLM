#include "chat_message.h"
#include <string.h>

ChatTemplate ChatTemplate::from_str(std::string template_str) {
  ChatTemplate t(template_str);
  return t;
}

std::string
ChatTemplate::render(std::vector<zinferlm::chat_message_t> messages) {
  return "";
}
