#pragma once

#include <bits/chrono.h>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
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

struct prefill_start_event_t {
  TimePoint timestamp = Clock::now();
  int tokens_count;
};

struct prefill_end_event_t {
  TimePoint timestamp = Clock::now();
  int tokens_count;
};

struct decode_start_event_t {
  TimePoint timestamp = Clock::now();
};

struct decode_end_event_t {
  TimePoint timestamp = Clock::now();
  int tokens_count;
};

using model_event_t =
    std::variant<token_generated_event_t, generation_completed_event_t,
                 prefill_start_event_t, prefill_end_event_t, decode_start_event_t>;

enum ModelEvent {
  TOKEN_GENERATED,
  GENERATION_COMPLETED,
  PREFILL_STARTED,
  PREFILL_COMPLETED,
  DECODE_STARTED,
  DECODE_COMPLETED,
};

using event_callback_t = std::function<void(zinferlm::events::model_event_t)>;

struct listener_t {
  event_callback_t callback;
  std::set<zinferlm::events::ModelEvent> events;
};

class EventDispatcher {
private:
  std::map<int, listener_t> listeners;
  int callbacks_count_ = 0;
  int callback_seq_ = 1;
  EventDispatcher() = default;

public:
  EventDispatcher(const EventDispatcher &) = delete;
  EventDispatcher(EventDispatcher &&) = delete;

  static EventDispatcher &get_instance();
  int listen(event_callback_t listener,
             std::set<zinferlm::events::ModelEvent> events);
  void unlisten(int observer_id);
  void dispatch(zinferlm::events::ModelEvent event_name,
                zinferlm::events::model_event_t event);
};

}; // namespace events
}; // namespace zinferlm
