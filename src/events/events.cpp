#include <mutex>
#include <utility>
#include <vector>
#include <zinferlm/events.h>

using EventDispatcher = zinferlm::events::EventDispatcher;

EventDispatcher &EventDispatcher::get_instance() {
  static EventDispatcher instance;
  return instance;
}

int EventDispatcher::listen(zinferlm::events::event_callback_t listener,
                            std::set<zinferlm::events::InferenceEvent> events) {
  std::lock_guard<std::mutex> lock(mutex_);
  int id = next_id_++;
  listeners.push_back(
      {.id = id, .callback = std::move(listener), .events = std::move(events)});
  return id;
}

void EventDispatcher::unlisten(int observer_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::erase_if(listeners,
                [observer_id](const zinferlm::events::listener_t &l) {
                  return l.id == observer_id;
                });
}

void EventDispatcher::dispatch(zinferlm::events::InferenceEvent event_name,
                               zinferlm::events::inference_event_t event) {
  // Snapshot the matching callbacks under lock, then invoke them outside the
  // lock so a callback may safely unlisten (or another thread may listen /
  // unlisten) without invalidating iteration.
  std::vector<zinferlm::events::event_callback_t> callbacks;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks.reserve(listeners.size());
    for (auto &it : listeners) {
      if (it.events.contains(event_name)) {
        callbacks.push_back(it.callback);
      }
    }
  }
  for (auto &cb : callbacks) {
    cb(event);
  }
}
