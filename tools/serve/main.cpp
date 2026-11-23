#include "server.hpp"

#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include <zinferlm/models.h>

#include "../log_flags.h"
#include "logging/log.h"
#include "zinferlm/chat.h"
#include "zinferlm/config.hpp"

int main(int argc, char *argv[]) {
  std::string ip = "0.0.0.0";
  unsigned short port = 8080;
  int num_threads = 4;
  std::string model_path = "";

  auto args = zinferlm_tools::parse_log_flags(argc, argv);
  if (!args.empty())
    model_path = args[0];

  zinferlm::config_t config;
  zinferlm::ConfigManager::init(config);

  LOG_INF("Loading model from file: %s\n", model_path.c_str());
  auto is_loaded = zinferlm::Model::load(model_path.c_str());
  if (!is_loaded) {
    LOG_ERR("Failed to load model: %s\n", model_path.c_str());
    return 1;
  }
  auto &model = zinferlm::Model::instance();
  zinferlm::model_info_t info = model.info();
  LOG_INF("Model Loaded: %s\n", info.name.c_str());

  zinferlm::Chat chat(model);

  if (num_threads < 1)
    num_threads = 1;

  Server server(ip, port, num_threads);
  server.run();

  chat.stop();
  if (!zinferlm::Model::unload()) {
    LOG_ERR("Model could not be unloaded: engine still running\n");
    return 1;
  }
  return 0;
}
