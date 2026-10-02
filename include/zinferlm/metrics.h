#pragma once

#include <bits/chrono.h>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

namespace zinferlm {
namespace metrics {
struct model_metrics_t {
  float time_to_first_token;
  float throughput_in_secs;
  long prefill_latency;
  float decode_latency;
};

class LLMMetricsCollector {
private:
  int observer_id_;
  TimePoint prefill_start_time_, prefill_end_time_;
  TimePoint decode_start_time_, decode_end_time_;
  int prefill_tokens_count_, decode_tokens_count_;

public:
  void collect();
  model_metrics_t get_metrics();
  void reset();
};
} // namespace metrics
} // namespace zinferlm
