#pragma once

#include <condition_variable>
#include <cstdint>
#include <future>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "zinferlm/export.h"
#include "zinferlm/models.h"

namespace zinferlm {

struct scheduler_config_t;

struct chat_params_t {
  std::string chat_id;
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

class ZINFERLM_API Chat {
private:
  zinferlm::Model &model_;
  bool debug_ = false;

public:
  Chat(zinferlm::Model &m) : model_(m) {}
  std::future<std::string> invoke(std::vector<ChatMessage> &messages,
                                  const zinferlm::chat_params_t &params);
  void set_debug(bool enabled) { debug_ = enabled; }

  // Stops the inference engine: cancels in-flight work, joins every engine
  // thread and releases engine resources. Must be called before
  // Model::unload(). Static because the engine is process-wide, even though a
  // Chat object is created per request.
  static void stop();
};

struct chat_task_t {
  std::vector<zinferlm::ChatMessage> messages;
  chat_params_t params;
};

} // namespace zinferlm
