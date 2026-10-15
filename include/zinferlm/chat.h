#pragma once

#include <string>
#include <zinferlm/models.h>

namespace zinferlm {

struct chat_params_t {
  uint32_t max_tokens;
  float temperature;
  uint32_t top_k;
  float top_p;
};

class ChatMessage {
public:
  std::string message;
  std::string role;
  ChatMessage(std::string r, std::string msg) : role(r), message(msg) {}
};
class SystemMessage : public ChatMessage {
public:
  SystemMessage(std::string msg) : ChatMessage("system", msg) {}
};
class UserMessage : public ChatMessage {
public:
  UserMessage(std::string msg) : ChatMessage("user", msg) {}
};
class AssistantMessage : public ChatMessage {
public:
  AssistantMessage(std::string msg) : ChatMessage("assistant", msg) {}
};

class Chat {
private:
  zinferlm::Model &model_;
  bool debug_ = false;

public:
  Chat(zinferlm::Model &m) : model_(m) {}
  ChatMessage invoke(std::vector<ChatMessage> &messages,
                     const zinferlm::chat_params_t &params);
  void set_debug(bool enabled) { debug_ = enabled; }
};
}; // namespace zinferlm
