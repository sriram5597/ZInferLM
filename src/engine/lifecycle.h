#pragma once

#include <atomic>

namespace zinferlm {
namespace internal {

// Process-wide engine liveness flag. Set once the inference engine owns live
// threads and model-derived state; cleared only after every engine thread has
// been joined and that state has been released.
//
// Lives in its own header so model code can consult it without pulling in the
// engine, inference and ggml layers.
inline std::atomic<bool> &engine_live_flag() {
  static std::atomic<bool> live{false};
  return live;
}

inline bool engine_is_running() {
  return engine_live_flag().load(std::memory_order_acquire);
}

inline void engine_mark_started() {
  engine_live_flag().store(true, std::memory_order_release);
}

inline void engine_mark_stopped() {
  engine_live_flag().store(false, std::memory_order_release);
}

} // namespace internal
} // namespace zinferlm
