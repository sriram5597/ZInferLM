#include <algorithm>
#include <utility>
#include <zinferlm/events.h>

using EventDispatcher = zinferlm::events::EventDispatcher;

EventDispatcher &EventDispatcher::get_instance() {
  static EventDispatcher instance;
  return instance;
}

int EventDispatcher::listen(zinferlm::events::event_callback_t listener,
                            std::set<zinferlm::events::ModelEvent> events) {
  int seq = callback_seq_;
  listeners.insert(
      {seq, {.callback = std::move(listener), .events = std::move(events)}});
  callback_seq_++;
  callbacks_count_++;
  return seq;
}

void EventDispatcher::unlisten(int observer_id) {
  listeners.erase(observer_id);
  callbacks_count_--;
}

void EventDispatcher::dispatch(zinferlm::events::ModelEvent event_name,
                               zinferlm::events::model_event_t event) {
  for (auto &it : listeners) {
    listener_t l = it.second;
    if (l.events.contains(event_name)) {
      l.callback(event);
    }
  }
}
