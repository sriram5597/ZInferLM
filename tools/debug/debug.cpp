#include <iostream>
#include <string>
#include <zinferlm/chat.h>
#include <zinferlm/models.h>

#include "../log_flags.h"
#include "logging/log.h"

int main(int argc, char *argv[]) {
  auto args = zinferlm_tools::parse_log_flags(argc, argv);
  if (args.empty()) {
    LOG_ERR("Usage: %s <model_path> [prompt] %s\n", argv[0],
            zinferlm_tools::log_flags_usage());
    return 1;
  }

  std::string model_path = args[0];
  std::string prompt = (args.size() >= 2) ? args[1] : "";

  auto is_loaded = zinferlm::Model::load(model_path.c_str());
  if (!is_loaded) {
    LOG_ERR("Failed to load model: %s\n", model_path.c_str());
    return 1;
  }

  auto &model = zinferlm::Model::instance();

  zinferlm::Chat chat(model);
  chat.set_debug(true);
  std::vector<zinferlm::ChatMessage> messages = {zinferlm::UserMessage("hi")};
  std::string output =
      chat.invoke(messages, {.max_tokens = 1, .temperature = 1.0}).get();
  std::cout << output << std::endl;

  zinferlm::Chat::stop();
  zinferlm::Model::unload();
  return 0;
}
