#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <zinferlm/events.h>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

#include "inference.h"
#include "kv_cache/cache.h"
#include "models/model_impl.h"
#include "sampling/samplers.h"

namespace zinferlm {
namespace internal {

using zinferlm::events::EventDispatcher;
using zinferlm::events::InferenceEvent;

Inference::Inference(KVCache *cache, zinferlm::Model &model)
    : model_(model), model_impl_(model.impl()), cache_(cache) {
  auto backend_ptrs = model_impl_.backends();
  sched_ = ggml_backend_sched_ptr{ggml_backend_sched_new(
      backend_ptrs.data(), nullptr, backend_ptrs.size(), 16384, false, true)};
  reserve_memory();
}

bool Inference::reserve_memory() {
  if (reserved) {
    return true;
  }

  zinferlm::model_config_t cfg = model_impl_.config();
  const uint32_t max_ctx = cfg.max_context_len;

  ggml_init_params params = {
      .mem_size = cfg.n_blocks * 256 * ggml_tensor_overhead(),
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  ggml_context *mctx = ggml_init(params);
  ggml_cgraph *mgf = ggml_new_graph(mctx);
  model_impl_.create_input_tensors(mctx, prefill_chunksize_, max_ctx);
  cache_->ensure_blocks(0, (int)(max_ctx + prefill_chunksize_));
  ggml_tensor *out = model_impl_.build_graph(mctx, mgf, cache_);
  ggml_build_forward_expand(mgf, out);

  for (ggml_tensor *t : model_impl_.get_input_tensors()) {
    ggml_backend_sched_set_tensor_backend(sched_.get(), t,
                                          model_impl_.get_backend());
  }

  bool ok = ggml_backend_sched_reserve(sched_.get(), mgf);
  ggml_free(mctx);
  cache_->reset();
  reserved = ok;
  return ok;
}

ggml_context *Inference::init_context(uint64_t n_op_estimate) {
  uint64_t ctx_size = n_op_estimate * ggml_tensor_overhead() +
                      ggml_graph_overhead_custom(n_op_estimate, false) + 1024;
  if (ctx_buffer_.size() < ctx_size) {
    ctx_buffer_.resize(ctx_size);
  }
  ggml_init_params params = {
      .mem_size = ctx_size, .mem_buffer = ctx_buffer_.data(), .no_alloc = true};
  if (ctx_) {
    ggml_free(ctx_);
  }
  ctx_ = ggml_init(params);
  return ctx_;
}

Inference::~Inference() {
  ggml_free(ctx_);
}

ggml_cgraph *Inference::new_graph(uint32_t n_tokens, uint32_t past_tokens) {
  gf_ = ggml_new_graph(ctx_);
  model_impl_.create_input_tensors(ctx_, n_tokens, past_tokens);
  return gf_;
}

void Inference::set_output(ggml_tensor *out) {
  output_ = out;
  ggml_build_forward_expand(gf_, output_);
}

bool Inference::prepare() {
  ggml_backend_sched_reset(sched_.get());
  for (ggml_tensor *t : model_impl_.get_input_tensors()) {
    ggml_backend_sched_set_tensor_backend(sched_.get(), t,
                                          model_impl_.get_backend());
  }

  if (!ggml_backend_sched_alloc_graph(sched_.get(), gf_)) {
    std::cerr << "Failed to allocate scheduler memory" << std::endl;
    return false;
  }
  return true;
}

ggml_tensor *Inference::compute(std::vector<int32_t> input, int past_tokens) {
  if (debug_) {
    debug_cb_data_ = {debug_, {}};
    ggml_backend_sched_set_eval_callback(sched_.get(), debug_eval_callback,
                                         &debug_cb_data_);
  } else {
    ggml_backend_sched_set_eval_callback(sched_.get(), nullptr, nullptr);
  }

  if (debug_) {
    std::cout << "Input Tokens: [ ";
    for (auto t : input) {
      std::cout << t << ", ";
    }
    std::cout << " ]\n";
  }

  std::vector<int32_t> lookup_rows = cache_->get_lookup_rows(0, past_tokens + input.size());
  std::vector<int64_t> write_rows = cache_->get_write_rows(0, past_tokens, input.size());
  std::cerr << "[debug] compute: blocks_count=" << cache_->blocks_count()
            << " lookup_rows.size=" << lookup_rows.size()
            << " write_rows.size=" << write_rows.size() << std::endl;
  for (auto r : write_rows) {
    if (r < 0 || r >= (int64_t)cache_->blocks_count() * 16) {
      std::cerr << "[debug] compute: BAD write index=" << r << std::endl;
    }
  }
  model_impl_.set_inputs(input, past_tokens, lookup_rows, write_rows);

  auto status = ggml_backend_sched_graph_compute(sched_.get(), gf_);
  if (status != GGML_STATUS_SUCCESS) {
    std::cerr << "Graph compute failed with status: " << status << std::endl;
    return nullptr;
  }
  return output_;
}

std::vector<float> Inference::predict(std::vector<int32_t> tokens,
                                      int past_tokens) {
  zinferlm::model_config_t cfg = model_impl_.config();
  const uint32_t n_tokens = (uint32_t)tokens.size();

  cache_->ensure_blocks(0, past_tokens + (int)n_tokens);

  init_context(cfg.n_blocks * 64 + 256);
  new_graph(n_tokens, past_tokens);
  ggml_tensor *output = model_impl_.build_graph(ctx_, gf_, cache_);
  set_output(output);
  if (!prepare()) {
    return {};
  }

  ggml_tensor *out = compute(tokens, past_tokens);
  GGML_ASSERT(tokens.size() <= out->ne[1]);

  std::vector<float> logits(out->ne[0]);
  uint64_t offset =
      (tokens.size() - 1) * out->ne[0] * ggml_type_size(out->type);
  ggml_backend_tensor_get(out, logits.data(), offset,
                          out->ne[0] * ggml_type_size(out->type));
  return logits;
}

std::string Inference::invoke(std::string input, uint32_t max_tokens) {
  EventDispatcher &dispatcher = EventDispatcher::get_instance();
  Tokenizer tokenizer = Tokenizer::for_model(model_);
  std::string output = "";
  dispatcher.dispatch(InferenceEvent::TOKENIZER_STARTED,
                      events::tokenizer_started_event_t{
                          .str_len = input.length(),
                      });
  std::vector<int32_t> tokens = tokenizer.tokenize(input);
  dispatcher.dispatch(InferenceEvent::TOKENIZER_COMPLETED,
                      events::tokenizer_completed_event_t{
                          .num_tokens = tokens.size(),
                      });
  int past_tokens = 0;
  std::cout << "Tokenizer completed" << std::endl;

  // ---- chunked prefill ----
  events::prefill_start_event_t p_start;
  p_start.tokens_count = tokens.size();
  dispatcher.dispatch(InferenceEvent::PREFILL_STARTED, p_start);

  std::vector<float> logits;
  for (size_t start = 0; start < tokens.size(); start += prefill_chunksize_) {
    size_t count = std::min<size_t>(prefill_chunksize_, tokens.size() - start);
    std::vector<int32_t> chunk(tokens.begin() + start,
                               tokens.begin() + start + count);
    logits = predict(chunk, past_tokens);
    past_tokens += (int)count;
  }

  events::prefill_end_event_t p_end;
  p_end.tokens_count = tokens.size();
  dispatcher.dispatch(InferenceEvent::PREFILL_COMPLETED, p_end);
  std::cout << "Prefill completed..." << std::endl;

  // ---- decode loop ----
  int i = 0;
  for (; i < (int)max_tokens; i++) {
    std::pair<uint64_t, float> sample = sampler_.sample(logits);
    std::string out_token =
        tokenizer.decode(static_cast<uint32_t>(sample.first));
    if (tokenizer.is_stop_token(out_token)) {
      dispatcher.dispatch(InferenceEvent::GENERATION_COMPLETED,
                          events::generation_completed_event_t{
                              .status = StreamStatus::EOS,
                              .num_tokens = static_cast<uint32_t>(i)});
      break;
    }

    output += out_token;

    dispatcher.dispatch(
        InferenceEvent::TOKEN_GENERATED,
        events::token_generated_event_t{.seq_id = static_cast<uint32_t>(i),
                                        .token = std::string_view{out_token}});

    std::vector<int32_t> next = {static_cast<int32_t>(sample.first)};
    logits = predict(next, past_tokens);
    past_tokens += 1;
  }
  if (i >= (int)max_tokens) {
    dispatcher.dispatch(InferenceEvent::GENERATION_COMPLETED,
                        events::generation_completed_event_t{
                            .status = StreamStatus::MAX_CTX_REACHED});
  }
  return output;
}

} // namespace internal
} // namespace zinferlm
