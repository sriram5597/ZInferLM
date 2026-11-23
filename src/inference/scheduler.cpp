#include "scheduler.hpp"
#include "ggml.h"
#include "kv_cache/cache.h"
#include "logging/log.h"
#include "sampling/samplers.h"
#include "worker/worker.hpp"
#include "zinferlm/config.hpp"
#include "zinferlm/events.h"
#include "zinferlm/models.h"
#include "zinferlm/tokenizer.h"
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <sys/stat.h>
#include <thread>
#include <utility>

namespace zinferlm {
namespace internal {

static const char *seq_status_str(SeqStatus status) {
  switch (status) {
  case SeqStatus::PENDING:
    return "PENDING";
  case SeqStatus::PREFILL:
    return "PREFILL";
  case SeqStatus::DECODE:
    return "DECODE";
  case SeqStatus::COMPLETED:
    return "COMPLETED";
  }
  return "UNKNOWN";
}

InferenceScheduler::InferenceScheduler() {
  zinferlm::model_config_t cfg = zinferlm::Model::instance().config();
  ModelImpl &model_impl = zinferlm::Model::instance().impl();
  zinferlm::config_t config = zinferlm::ConfigManager::get_config();
  prefill_chunk_size_ = config.prefill_chunk_size;
  max_batches_ = (int)config.max_batches;
  cache_ = std::make_unique<KVCache>(
      model_impl.get_backend(), config.max_ctx, cfg.n_blocks, cfg.nkv,
      cfg.embedding_dim / cfg.nheads, GGML_TYPE_F16);
  runtime_ =
      std::make_unique<Inference>(cache_.get(), zinferlm::Model::instance(),
                                  max_batches_, prefill_chunk_size_);
  active_seq_count_ = 0;
  total_tokens_in_batch_ = 0;
  seq_list_.resize(max_batches_);
  sched_thread_ = std::jthread([this](std::stop_token tok) { this->run(tok); });
  tokenizer_ = Tokenizer::for_model(zinferlm::Model::instance());
}

InferenceScheduler::~InferenceScheduler() {
  stop();
  release_state();
}

void InferenceScheduler::run(std::stop_token tok) {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(seq_lock_);
      cv_.wait(lock, tok, [this] {
        return !pending_queue_.empty() || active_seq_count_ > 0;
      });
    }
    if (tok.stop_requested()) {
      {
        std::lock_guard<std::mutex> lock(seq_lock_);
        LOG_INF("scheduler stopping: active=%d pending=%zu\n",
                active_seq_count_, pending_queue_.size());
      }
      // Teardown happens in release_state(), called by the owner after the
      // join, so nothing is freed while this thread is still inside a step.
      break;
    }
    step_();
  }
}

void InferenceScheduler::stop() {
  accepting_.store(false, std::memory_order_release);
  sched_thread_.request_stop();
  cv_.notify_all();
  if (sched_thread_.joinable() &&
      sched_thread_.get_id() != std::this_thread::get_id()) {
    sched_thread_.join();
  }
}

bool InferenceScheduler::on_scheduler_thread() const {
  return sched_thread_.get_id() == std::this_thread::get_id();
}

void InferenceScheduler::release_state() {
  std::vector<std::pair<std::string, std::promise<std::string>>> cancelled;
  {
    std::lock_guard<std::mutex> lock(seq_lock_);
    auto take = [&cancelled](seq_info_t &info) {
      cancelled.emplace_back(info.seq_id, std::move(info.promise));
    };
    while (!pending_queue_.empty()) {
      take(pending_queue_.front());
      pending_queue_.pop();
    }
    for (auto &slot : seq_list_) {
      if (slot.has_value()) {
        take(*slot);
        slot = std::nullopt;
      }
    }
    active_slots_.clear();
    active_seq_count_ = 0;
    total_tokens_in_batch_ = 0;
  }

  for (auto &[seq_id, promise] : cancelled) {
    LOG_WRN("sequence cancelled: seq_id=%s reason=engine shutdown\n",
            seq_id.c_str());
    events::EventDispatcher::get_instance().dispatch(
        seq_id, events::InferenceEvent::GENERATION_COMPLETED,
        events::generation_completed_event_t{.status = StreamStatus::CANCELLED,
                                             .num_tokens = 0});
    promise.set_exception(
        std::make_exception_ptr(worker_stopped("engine shutting down")));
  }

  // Free everything that was derived from the model while the model is still
  // loaded: the backend scheduler and the KV cache buffer both reference
  // backends owned by ModelImpl.
  runtime_.reset();
  cache_.reset();
}

void InferenceScheduler::schedule_sequence_() {
  std::lock_guard<std::mutex> lock(seq_lock_);
  int slot_id = -1;
  int admitted = 0;
  while (active_seq_count_ < max_batches_ && !pending_queue_.empty()) {
    for (int i = 0; i < max_batches_; i++) {
      if (!seq_list_[i].has_value()) {
        slot_id = i;
        break;
      }
    }
    if (slot_id < 0) {
      LOG_DBG("batch full: active=%d/%d pending=%zu\n", active_seq_count_,
              max_batches_, pending_queue_.size());
      return;
    }
    seq_list_[slot_id] = std::move(pending_queue_.front());
    seq_list_[slot_id]->status = SeqStatus::PREFILL;
    pending_queue_.pop();
    ++active_seq_count_;
    ++admitted;
    LOG_INF("sequence scheduled: seq_id=%s slot=%d (PREFILL)\n",
            seq_list_[slot_id]->seq_id.c_str(), slot_id);
  }
  if (!pending_queue_.empty()) {
    LOG_DBG("admission done: admitted=%d active=%d/%d pending=%zu\n", admitted,
            active_seq_count_, max_batches_, pending_queue_.size());
  }
}

void InferenceScheduler::build_causal_mask_(model_input_t &input,
                                            size_t total_filled) {
  size_t start_row = 0;
  size_t start_col = 0;
  std::fill(input.causal_mask.begin(), input.causal_mask.end(),
            ggml_fp32_to_fp16(-INFINITY));
  for (auto slot_id : active_slots_) {
    const seq_info_t &info = seq_list_[slot_id].value();
    const size_t filled = info.past_tokens + info.seq_len;
    for (size_t row = 0; row < info.seq_len; row++) {
      const size_t pos = info.past_tokens + row;
      for (size_t col = 0; col < filled; col++) {
        if (col > pos) {
          break;
        }
        const size_t ind = (start_row + row) * total_filled + start_col + col;
        input.causal_mask[ind] = 0;
      }
    }
    start_row += info.seq_len;
    start_col += filled;
  }
}

void InferenceScheduler::build_input_(int slot_id, model_input_t &input) {
  seq_info_t &info = seq_list_[slot_id].value();
  const size_t len = info.inp_tokens.size();
  if (info.past_tokens < len) {
    info.status = SeqStatus::PREFILL;
    const size_t remaining = len - info.past_tokens;
    const size_t chunk_size = std::min(prefill_chunk_size_, remaining);
    input.tokens.insert(
        input.tokens.end(), info.inp_tokens.begin() + info.past_tokens,
        info.inp_tokens.begin() + info.past_tokens + chunk_size);
    info.seq_len = chunk_size;
    LOG_DBG("prefill chunk: seq_id=%s chunk=%zu remaining=%zu\n",
            info.seq_id.c_str(), chunk_size, remaining - chunk_size);
    for (size_t i = 0; i < chunk_size; i++) {
      input.positions.push_back(info.past_tokens + i);
    }
  } else {
    if (info.status != SeqStatus::DECODE) {
      LOG_INF("prefill completed: seq_id=%s prompt_tokens=%zu, entering "
              "DECODE\n",
              info.seq_id.c_str(), len);
    }
    info.status = SeqStatus::DECODE;
    input.tokens.push_back(info.out_tokens.back());
    input.positions.push_back(info.past_tokens);
    info.seq_len = 1;
  }
  if (input.offsets.empty()) {
    input.offsets.emplace_back(info.seq_len - 1);
  } else {
    input.offsets.emplace_back(input.offsets.back() + info.seq_len);
  }
}

void InferenceScheduler::complete_sequence_(int slot_id) {
  GGML_ASSERT(slot_id >= 0 && slot_id < max_batches_);
  GGML_ASSERT(seq_list_[slot_id].has_value());
  seq_info_t &info = seq_list_[slot_id].value();
  LOG_INF(
      "sequence finished: seq_id=%s slot=%d output_chars=%zu active=%d/%d\n",
      info.seq_id.c_str(), slot_id, info.out_str.size(), active_seq_count_ - 1,
      max_batches_);
  total_tokens_in_batch_ -= info.past_tokens;
  info.promise.set_value(info.out_str);
  seq_list_[slot_id] = std::nullopt;
  --active_seq_count_;
  auto it = std::find(active_slots_.begin(), active_slots_.end(), slot_id);
  if (it != active_slots_.end()) {
    active_slots_.erase(it);
  }
}

void InferenceScheduler::init_cache_info_(int slot_id, model_input_t &input) {
  auto &info = seq_list_[slot_id].value();
  const size_t filled = info.past_tokens + info.seq_len;
  auto lookup_rows = cache_->get_lookup_rows(slot_id, filled);
  auto write_rows =
      cache_->get_write_rows(slot_id, info.past_tokens, info.seq_len);
  input.cache_lookup_rows.insert(input.cache_lookup_rows.end(),
                                 lookup_rows.begin(), lookup_rows.end());
  input.cache_write_rows.insert(input.cache_write_rows.end(),
                                write_rows.begin(), write_rows.end());
}

void InferenceScheduler::step_() {
  schedule_sequence_();
  auto &dispatcher = events::EventDispatcher::get_instance();
  if (active_seq_count_ == 0) {
    return;
  }
  active_slots_.clear();
  for (int i = 0; i < max_batches_; i++) {
    if (seq_list_[i].has_value()) {
      active_slots_.push_back(i);
    }
  }
  model_input_t input;
  size_t total_tokens = 0;
  size_t total_filled = 0;
  for (auto slot : active_slots_) {
    build_input_(slot, input);
    cache_->ensure_blocks(slot, seq_list_[slot]->past_tokens +
                                    seq_list_[slot]->seq_len);
    init_cache_info_(slot, input);
    total_tokens += seq_list_[slot]->seq_len;
    total_filled += seq_list_[slot]->past_tokens + seq_list_[slot]->seq_len;
  }
  total_tokens_in_batch_ += total_filled;
  input.causal_mask.resize(total_tokens * total_filled);
  build_causal_mask_(input, total_filled);
  LOG_DBG("step: batch=%zu tokens=%zu filled=%zu\n", active_slots_.size(),
          total_tokens, total_filled);
  for (auto slot : active_slots_) {
    const seq_info_t &info = seq_list_[slot].value();
    LOG_DBG("step slot: slot=%d seq=%s status=%s seq_len=%zu past=%zu\n", slot,
            info.seq_id.c_str(), seq_status_str(info.status), info.seq_len,
            info.past_tokens);
  }
  const auto t_start = std::chrono::steady_clock::now();
  std::pair<std::vector<float>, int64_t> res =
      runtime_->predict(input, total_tokens, total_filled);
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::steady_clock::now() - t_start;
  LOG_DBG("step compute: tokens=%zu elapsed_ms=%.2f\n", total_tokens,
          elapsed.count());
  std::vector<int> finished_slots;
  for (size_t i = 0; i < active_slots_.size(); i++) {
    int slot_id = active_slots_[i];
    auto &current_seq = seq_list_[slot_id];
    auto sample_config = seq_list_[slot_id]->sample_config;
    Sampler sampler(sample_config);
    std::span<const float> logits(res.first.data() + i * res.second,
                                  res.second);
    auto sample = sampler.sample(logits);
    current_seq->past_tokens += current_seq->seq_len;
    std::string token = tokenizer_.decode(sample.first);
    LOG_DBG("token decoded: seq_id=%s pos=%zu token_id=%" PRIu64 " prob=%.4f\n",
            current_seq->seq_id.c_str(), current_seq->past_tokens - 1,
            sample.first, sample.second);
    if (tokenizer_.is_stop_token(token) ||
        current_seq->past_tokens >= current_seq->max_tokens) {
      StreamStatus status = StreamStatus::MAX_CTX_REACHED;
      if (tokenizer_.is_stop_token(token)) {
        status = StreamStatus::EOS;
      }
      LOG_INF("generation completed: seq_id=%s tokens=%zu reason=%s\n",
              current_seq->seq_id.c_str(), current_seq->past_tokens,
              status == StreamStatus::EOS ? "EOS" : "MAX_TOKENS");
      dispatcher.dispatch(
          current_seq->seq_id, events::InferenceEvent::GENERATION_COMPLETED,
          events::generation_completed_event_t{
              .status = status, .num_tokens = current_seq->past_tokens});
      finished_slots.push_back(slot_id);
    } else if (current_seq->past_tokens >= current_seq->inp_tokens.size()) {
      dispatcher.dispatch(
          current_seq->seq_id, events::InferenceEvent::TOKEN_GENERATED,
          events::token_generated_event_t{.seq_id = current_seq->seq_id,
                                          .token = token,
                                          .pos = current_seq->past_tokens - 1});
      current_seq->out_tokens.push_back(sample.first);
      current_seq->out_str += token;
    }
  }
  for (int slot_id : finished_slots) {
    complete_sequence_(slot_id);
  }
}

void InferenceScheduler::submit(seq_info_t &&seq) {
  std::optional<std::promise<std::string>> rejected;
  {
    std::lock_guard<std::mutex> lock(seq_lock_);
    if (!accepting_.load(std::memory_order_acquire)) {
      rejected = std::move(seq.promise);
    } else {
      LOG_INF("sequence queued: seq_id=%s prompt_tokens=%zu\n",
              seq.seq_id.c_str(), seq.inp_tokens.size());
      pending_queue_.push(std::move(seq));
    }
  }
  if (rejected.has_value()) {
    LOG_WRN("sequence rejected, engine is stopping\n");
    rejected->set_exception(
        std::make_exception_ptr(worker_stopped("engine is shutting down")));
    return;
  }
  cv_.notify_one();
}
} // namespace internal
} // namespace zinferlm
