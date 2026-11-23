#include <memory>
#include <mutex>
#include <stdexcept>
#include <zinferlm/config.hpp>

namespace zinferlm {
void ConfigManager::init(zinferlm::config_t cfg) {
  std::call_once(init_flag_,
                 [&]() { instance_.reset(new ConfigManager(cfg)); });
}

const config_t &ConfigManager::get_config() {
  if (instance_ == nullptr) {
    throw std::runtime_error("Config is accessed before init");
  }
  return instance_->config_;
}
} // namespace zinferlm
