#include <mutex>
#include <utility>
#include <vector>
#include <zinferlm/events.h>

using EventDispatcher = zinferlm::events::EventDispatcher;

EventDispatcher &EventDispatcher::get_instance() {
  static EventDispatcher instance;
  return instance;
}

void EventDispatcher::listen(
    std::string group_id, zinferlm::events::event_callback_t listener,
    std::set<zinferlm::events::InferenceEvent> events) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (listeners.find(group_id) != listeners.end()) {
    listeners[group_id].push_back(
        {.callback = std::move(listener), .events = std::move(events)});
  } else {
    listeners.insert(
        {group_id,
         {{.callback = std::move(listener), .events = std::move(events)}}});
  }
  callbacks_count_++;
}

void EventDispatcher::unlisten(std::string group_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  int callbacks = listeners[group_id].size();
  listeners.erase(group_id);
  callbacks_count_ -= callbacks;
}

void EventDispatcher::dispatch(std::string group_id,
                               zinferlm::events::InferenceEvent event_name,
                               zinferlm::events::inference_event_t event) {
  // Snapshot the matching callbacks under lock, then invoke them outside the
  // lock so a callback may safely unlisten (or another thread may listen /
  // unlisten) without invalidating iteration.
  std::vector<zinferlm::events::event_callback_t> callbacks;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks.reserve(listeners.size());
    for (auto &it : listeners[group_id]) {
      if (it.events.contains(event_name)) {
        callbacks.push_back(it.callback);
      }
    }
  }
  for (auto &cb : callbacks) {
    cb(event);
  }
}
