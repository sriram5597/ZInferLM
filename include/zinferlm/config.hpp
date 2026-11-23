#pragma once
#include <cstddef>
#include <memory>
#include <mutex>

namespace zinferlm {
struct config_t {
  size_t prefill_chunk_size = 512;
  size_t max_tokens = 4096;
  size_t max_ctx = 8192;
  size_t max_batches = 16;
};

class ConfigManager {
private:
  const config_t config_;

  inline static std::unique_ptr<ConfigManager> instance_ = nullptr;
  inline static std::once_flag init_flag_;

  explicit ConfigManager(config_t cfg) noexcept : config_(std::move(cfg)) {}

  ConfigManager(const ConfigManager &) = delete;
  ConfigManager &operator=(const ConfigManager &) = delete;
  ConfigManager(ConfigManager &&) = delete;
  ConfigManager &operator=(ConfigManager &&) = delete;

public:
  static void init(config_t config);
  static const config_t &get_config();
};
} // namespace zinferlm
