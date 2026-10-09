#include <chrono>
#include <iostream>
#include <variant>
#include <zinferlm/events.h>
#include <zinferlm/metrics.h>

using InferenceEvent = zinferlm::events::InferenceEvent;
using inference_event_t = zinferlm::events::inference_event_t;

void zinferlm::metrics::LLMMetricsCollector::collect() {
  events::event_callback_t collector =
      [this](events::inference_event_t event) -> void {
    if (auto p_start = std::get_if<events::prefill_start_event_t>(&event)) {
      this->prefill_start_time_ = p_start->timestamp;
      this->prefill_tokens_count_ = p_start->tokens_count;
    }
    if (auto p_end = std::get_if<events::prefill_end_event_t>(&event)) {
      this->prefill_end_time_ = p_end->timestamp;
    }
    if (auto tok_gen = std::get_if<events::token_generated_event_t>(&event)) {
      if (tok_gen->seq_id == 0) {
        this->first_token_generated_time_ = tok_gen->timestamp;
      }
    }
    if (auto t_start = std::get_if<events::tokenizer_started_event_t>(&event)) {
      this->tokenizer_start_time_ = t_start->timestamp;
    }
    if (auto t_end = std::get_if<events::tokenizer_completed_event_t>(&event)) {
      this->tokenizer_start_time_ = t_end->timestamp;
      this->inp_tokens_ = t_end->num_tokens;
    }
    if (auto gen_completed =
            std::get_if<events::GENERATION_COMPLETED>(&event)) {
      this->generation_completed_time_t = gen_completed->timestamp;
      this->out_tokens_ = gen_completed->num_tokens;
    }
  };
  events::EventDispatcher &dispatcher = events::EventDispatcher::get_instance();
  std::set<events::InferenceEvent> events = {
      events::InferenceEvent::TOKENIZER_STARTED,
      events::InferenceEvent::TOKENIZER_COMPLETED,
      events::InferenceEvent::PREFILL_STARTED,
      events::InferenceEvent::PREFILL_COMPLETED,
      events::InferenceEvent::GENERATION_COMPLETED,
      events::InferenceEvent::TOKEN_GENERATED};
  observer_id_ = dispatcher.listen(collector, events);
}

zinferlm::metrics::model_metrics_t
zinferlm::metrics::LLMMetricsCollector::get_metrics() {
  zinferlm::metrics::model_metrics_t metrics;
  metrics.prefill_latency =
      std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end_time_ -
                                                            prefill_start_time_)
          .count();
  metrics.time_to_first_token =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          first_token_generated_time_ - tokenizer_start_time_)
          .count();
  metrics.decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            generation_completed_time_t - prefill_end_time_)
                            .count();
  metrics.throughput_in_secs = (out_tokens_ * 1.0f) / (metrics.decode_time / 1000.0);
  metrics.total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                           generation_completed_time_t - tokenizer_start_time_)
                           .count();
  metrics.out_tokens = out_tokens_;
  metrics.inp_tokens = inp_tokens_;
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
