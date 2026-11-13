#include "inference/inference.h"
#include "kv_cache/cache.h"
#include "models/model_impl.h"
#include "sampling/samplers.h"
#include "zinferlm/tokenizer.h"
#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <optional>
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
};

class InferenceScheduler {
private:
  std::jthread sched_thread_;
  std::condition_variable_any cv_;

  std::mutex seq_lock_;
  std::queue<seq_info_t> pending_queue_;
  std::vector<std::optional<seq_info_t>> seq_list_;
  std::vector<int> active_slots_;
  int active_seq_count_;
  int max_batches_;
  uint64_t prefill_chunk_size_;

  std::unique_ptr<KVCache> cache_;
  std::unique_ptr<Inference> runtime_;
  Tokenizer tokenizer_;

  explicit InferenceScheduler();

  void schedule_sequence_();
  void step_();
  void build_causal_mask_(model_input_t &input, size_t total_filled);
  void build_input_(int slot_id, model_input_t &inp);
  void complete_sequence_(int slot_id);
  void init_cache_info_(int slot_id, model_input_t& input);

public:
  static InferenceScheduler &instance();
  void run(std::stop_token tok);
  void submit(seq_info_t &&info);
  void stop();
};
} // namespace internal
} // namespace zinferlm
