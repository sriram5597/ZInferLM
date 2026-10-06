#pragma once

#include "zinferlm/models.h"
#include <string>

namespace zinferlm {
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

public:
  Chat(zinferlm::Model &m) : model_(m) {}
  ChatMessage invoke(std::vector<ChatMessage> &messages);
};
}; // namespace zinferlm
