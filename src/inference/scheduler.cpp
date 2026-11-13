#include "scheduler.hpp"
#include "ggml.h"
#include "kv_cache/cache.h"
#include "sampling/samplers.h"
#include "zinferlm/models.h"
#include "zinferlm/tokenizer.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <thread>
#include <utility>

namespace zinferlm {
namespace internal {
InferenceScheduler::InferenceScheduler() {
  zinferlm::model_config_t cfg = zinferlm::Model::instance().config();
  ModelImpl &model_impl = zinferlm::Model::instance().impl();
  cache_ = std::make_unique<KVCache>(
      model_impl.get_backend(), cfg.max_context_len, cfg.n_blocks, cfg.nkv,
      cfg.embedding_dim / cfg.nheads, GGML_TYPE_F16);
  runtime_ =
      std::make_unique<Inference>(cache_.get(), zinferlm::Model::instance());
  max_batches_ = 16;
  prefill_chunk_size_ = 16;
  active_seq_count_ = 0;
  seq_list_.resize(max_batches_);
  sched_thread_ = std::jthread([this](std::stop_token tok) { this->run(tok); });
  tokenizer_ = Tokenizer::for_model(zinferlm::Model::instance());
}

InferenceScheduler &InferenceScheduler::instance() {
  static InferenceScheduler scheduler;
  return scheduler;
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
      active_seq_count_ = 0;
      break;
    }
    step_();
  }
}

void InferenceScheduler::stop() {
  sched_thread_.request_stop();
  if (sched_thread_.joinable()) {
    sched_thread_.join();
  }
}

void InferenceScheduler::schedule_sequence_() {
  std::lock_guard<std::mutex> lock(seq_lock_);
  int slot_id = -1;
  while (active_seq_count_ < max_batches_ && !pending_queue_.empty()) {
    for (int i = 0; i < max_batches_; i++) {
      if (!seq_list_[i].has_value()) {
        slot_id = i;
        break;
      }
    }
    if (slot_id < 0) {
      return;
    }
    seq_list_[slot_id] = std::move(pending_queue_.front());
    seq_list_[slot_id]->status = SeqStatus::PREFILL;
    pending_queue_.pop();
    ++active_seq_count_;
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
    input.tokens.insert(input.tokens.end(),
                        info.inp_tokens.begin() + info.past_tokens,
                        info.inp_tokens.begin() + info.past_tokens + chunk_size);
    info.seq_len = chunk_size;
    for (size_t i = 0; i < chunk_size; i++) {
      input.positions.push_back(info.past_tokens + i);
    }
  } else {
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
  seq_info_t &info = seq_list_[active_slots_[slot_id]].value();
  active_slots_.erase(active_slots_.begin() + slot_id);
  info.promise.set_value(info.out_str);
  seq_list_[slot_id] = std::nullopt;
  --active_seq_count_;
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
  input.causal_mask.resize(total_tokens * total_filled);
  build_causal_mask_(input, total_filled);
  std::pair<std::vector<float>, int64_t> res =
      runtime_->predict(input, total_tokens, total_filled);
  for (int i = 0; i < active_slots_.size(); i++) {
    int slot_id = active_slots_[i];
    auto &current_seq = seq_list_[slot_id];
    auto sample_config = seq_list_[slot_id]->sample_config;
    Sampler sampler(sample_config);
    std::span<const float> logits(res.first.data() + i * res.second,
                                  res.second);
    auto sample = sampler.sample(logits);
    current_seq->past_tokens += current_seq->seq_len;
    std::string token = tokenizer_.decode(sample.first);
    if (tokenizer_.is_stop_token(token)) {
      complete_sequence_(slot_id);
    } else if (current_seq->past_tokens >= current_seq->inp_tokens.size()) {
      current_seq->out_tokens.push_back(sample.first);
      current_seq->out_str += token;
    }
  }
}

void InferenceScheduler::submit(seq_info_t &&seq) {
  std::lock_guard<std::mutex> lock(seq_lock_);
  pending_queue_.push(std::move(seq));
  cv_.notify_one();
}
} // namespace internal
} // namespace zinferlm
