#include "inference/inference.h"
#include "kv_cache/cache.h"
#include "models/model_impl.h"
#include "sampling/samplers.h"
#include "zinferlm/tokenizer.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <optional>
#include <prometheus/counter.h>
#include <prometheus/family.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace zinferlm {
namespace internal {

enum SeqStatus {
  PENDING,
  PREFILL,
  DECODE,
  COMPLETED,
};

struct seq_info_t {
  std::string seq_id;
  size_t past_tokens;
  size_t seq_len;
  SeqStatus status;
  size_t max_tokens;
  sampler_params_t sample_config;
  std::promise<std::string> promise;
  std::vector<int32_t> inp_tokens;
  std::vector<int32_t> out_tokens;
  std::string out_str;
  std::chrono::steady_clock::time_point scheduled_time;
  std::chrono::steady_clock::time_point last_token_time;
};

struct scheduler_metrics_t {
  prometheus::Family<prometheus::Counter> *total_input_tokens;
  prometheus::Family<prometheus::Counter> *total_tokens_generated;
  prometheus::Histogram *ttft;
  prometheus::Histogram *itl;
  prometheus::Histogram* request_duration;
  prometheus::Family<prometheus::Counter> *total_requests;
  prometheus::Family<prometheus::Gauge> *active_sequence;
  prometheus::Family<prometheus::Gauge> *pending_requests;
};

class InferenceScheduler {
private:
  friend class Engine;

  std::jthread sched_thread_;
  std::condition_variable_any cv_;

  std::mutex seq_lock_;
  std::queue<seq_info_t> pending_queue_;
  std::vector<std::optional<seq_info_t>> seq_list_;
  std::vector<int> active_slots_;
  int active_seq_count_;
  int max_batches_ = 16;
  uint64_t prefill_chunk_size_;
  size_t total_tokens_in_batch_;
  std::atomic<bool> accepting_{true};

  std::unique_ptr<KVCache> cache_;
  std::unique_ptr<Inference> runtime_;
  Tokenizer tokenizer_;
  scheduler_metrics_t metrics_{};

  explicit InferenceScheduler();

  void schedule_sequence_();
  void step_();
  void build_causal_mask_(model_input_t &input, size_t total_filled);
  void build_input_(int slot_id, model_input_t &inp);
  void complete_sequence_(int slot_id);
  void init_cache_info_(int slot_id, model_input_t &input);
  void init_metrics();

public:
  ~InferenceScheduler();

  void run(std::stop_token tok);
  void submit(seq_info_t &&info);

  // Stops the scheduler thread and joins it. New submissions are rejected.
  void stop();

  // Fails every active and queued sequence, then releases all state derived
  // from the model (compute graph, KV cache buffer). Must only run after
  // stop(), never concurrently with the scheduler thread.
  void release_state();

  // True when the calling thread is the scheduler thread.
  bool on_scheduler_thread() const;
};
} // namespace internal
} // namespace zinferlm
