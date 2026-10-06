#pragma once

#include <bits/chrono.h>
#include <cstdint>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

namespace zinferlm {
namespace metrics {
struct model_metrics_t {
  float time_to_first_token;
  float throughput_in_secs;
  long prefill_latency;
  float decode_latency;
  uint32_t out_tokens;
  uint32_t inp_tokens;
  long total_time;
  long decode_time;
};

class LLMMetricsCollector {
private:
  int observer_id_;
  TimePoint prefill_start_time_, prefill_end_time_;
  TimePoint decode_start_time_, decode_end_time_;
  TimePoint tokenizer_start_time_, tokenizer_end_time_;
  TimePoint first_token_generated_time_, generation_completed_time_t;
  int prefill_tokens_count_, decode_tokens_count_;
  uint32_t out_tokens_, inp_tokens_;

public:
  void collect();
  model_metrics_t get_metrics();
  void reset();
};
} // namespace metrics
} // namespace zinferlm
