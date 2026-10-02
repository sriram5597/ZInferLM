#pragma once

#include <bits/chrono.h>
#include <cstdint>
#include <functional>
#include <map>
#include <string_view>
#include <unordered_map>

#include <variant>
#include <zinferlm/models.h>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

namespace zinferlm {
namespace events {
struct token_generated_event_t {
  uint32_t seq_id;
  std::string_view token;
  TimePoint timestamp = Clock::now();
};

struct generation_completed_event_t {
  TimePoint timestamp = Clock::now();
  zinferlm::StreamStatus status;
};

using model_event_t = std::variant<token_generated_event_t, generation_completed_event_t>;

enum ModelEvent {
  TOKEN_GENERATED,
  GENERATION_COMPLETED,
};

using model_event_listener_t =
    std::function<void(zinferlm::events::model_event_t)>;

class EventDispatcher {
private:
  std::unordered_map<zinferlm::events::ModelEvent,
                     std::map<int, model_event_listener_t>>
      model_callbacks_;
  int callbacks_count_ = 0;
  int callback_seq_ = 1;
  EventDispatcher() = default;

public:
  EventDispatcher(const EventDispatcher &) = delete;
  EventDispatcher(EventDispatcher &&) = delete;

  static EventDispatcher &get_instance();
  int listen(zinferlm::events::ModelEvent e, model_event_listener_t listener);
  void unlisten(zinferlm::events::ModelEvent e, int observer_id);
  void dispatch(zinferlm::events::ModelEvent event_name,
                zinferlm::events::model_event_t event);
};

}; // namespace events
}; // namespace zinferlm
