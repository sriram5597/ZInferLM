#include <zinferlm/events.h>

using EventDispatcher = zinferlm::events::EventDispatcher;

EventDispatcher &EventDispatcher::get_instance() {
  static EventDispatcher instance;
  return instance;
}

int EventDispatcher::listen(zinferlm::events::ModelEvent event,
                            model_event_listener_t listener) {
  auto &callbacks = model_callbacks_[event];
  int seq = callback_seq_;
  callbacks.insert({seq, std::move(listener)});
  callback_seq_++;
  callbacks_count_++;
  return seq;
}

void EventDispatcher::unlisten(zinferlm::events::ModelEvent event,
                               int observer_id) {
  auto &callbacks = model_callbacks_[event];
  callbacks.erase(observer_id);
  callbacks_count_--;
}

void EventDispatcher::dispatch(zinferlm::events::ModelEvent event_name,
                               zinferlm::events::model_event_t event) {
  auto &callbacks = model_callbacks_[event_name];
  for (auto &it : callbacks) {
    model_event_listener_t cb = it.second;
    cb(event);
  }
}
