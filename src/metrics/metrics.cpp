#include <chrono>
#include <iostream>
#include <variant>
#include <zinferlm/events.h>
#include <zinferlm/metrics.h>

using ModelEvent = zinferlm::events::ModelEvent;
using model_event_t = zinferlm::events::model_event_t;

void zinferlm::metrics::LLMMetricsCollector::collect() {
  events::event_callback_t collector =
      [this](events::model_event_t event) -> void {
    if (auto p_start = std::get_if<events::prefill_start_event_t>(&event)) {
      this->prefill_start_time_ = p_start->timestamp;
      this->prefill_tokens_count_ = p_start->tokens_count;
    }
    if (auto p_end = std::get_if<events::prefill_end_event_t>(&event)) {
      this->prefill_end_time_ = p_end->timestamp;
    }
    if (auto d_start = std::get_if<events::decode_start_event_t>(&event)) {
      this->decode_start_time_ = d_start->timestamp;
    }
  };
  events::EventDispatcher &dispatcher = events::EventDispatcher::get_instance();
  std::set<events::ModelEvent> events = {
      events::ModelEvent::PREFILL_STARTED,
      events::ModelEvent::PREFILL_COMPLETED,
      events::ModelEvent::DECODE_STARTED,
      events::ModelEvent::DECODE_COMPLETED,
  };
  observer_id_ = dispatcher.listen(collector, events);
}

zinferlm::metrics::model_metrics_t
zinferlm::metrics::LLMMetricsCollector::get_metrics() {
  zinferlm::metrics::model_metrics_t metrics;
  metrics.prefill_latency =
      std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end_time_ -
                                                            prefill_start_time_).count();
  return metrics;
}

void zinferlm::metrics::LLMMetricsCollector::reset() {
  prefill_start_time_ = TimePoint{};
  prefill_end_time_ = TimePoint{};
  decode_start_time_ = TimePoint{};
  decode_end_time_ = TimePoint{};
  prefill_tokens_count_ = 0;
  decode_tokens_count_ = 0;
}
