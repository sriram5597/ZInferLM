#include <mutex>
#include <utility>
#include <vector>
#include <zinferlm/events.h>

using EventDispatcher = zinferlm::events::EventDispatcher;

EventDispatcher &EventDispatcher::get_instance() {
  static EventDispatcher instance;
  return instance;
}

int EventDispatcher::listen(std::string listener_id,
                            zinferlm::events::event_callback_t listener,
                            std::set<zinferlm::events::InferenceEvent> events, bool capture_all_events) {
  std::lock_guard<std::mutex> lock(mutex_);
  int id = next_id_++;
  listeners_.emplace(listener_id, listener_t{.id = listener_id,
                                             .capture_all_events=capture_all_events,
                                             .callback = std::move(listener), .events = std::move(events)});
  return id;
}

void EventDispatcher::unlisten(std::string listener_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  listeners_.erase(listener_id);
}

void EventDispatcher::dispatch(std::string source_id, zinferlm::events::InferenceEvent event_name,
                               zinferlm::events::inference_event_t event) {
  // Snapshot the matching callbacks under lock, then invoke them outside the
  // lock so a callback may safely unlisten (or another thread may listen /
  // unlisten) without invalidating iteration.
  std::vector<zinferlm::events::event_callback_t> callbacks;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks.reserve(listeners_.size());
    for (auto &it : listeners_) {
      if (it.second.events.contains(event_name)) {
        if (!it.second.capture_all_events && it.second.id != source_id) {
          continue;
        }
        callbacks.push_back(it.second.callback);
      }
    }
  }
  for (auto &cb : callbacks) {
    cb(event);
  }
}
