#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
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
#include "sampling/samplers.h"

using zinferlm::events::EventDispatcher;
using zinferlm::events::InferenceEvent;

namespace zinferlm {

Inference::Inference(Model &model) : model_(model) {
  auto backend_ptrs = model_.backends();
  sched_ = ggml_backend_sched_ptr{ggml_backend_sched_new(
      backend_ptrs.data(), nullptr, backend_ptrs.size(), 16384, false, true)};

  model_config_t cfg = model_.config();
  cache_ = std::make_unique<KVCache>(model_.get_backend(), cfg.max_context_len,
                                     cfg.n_blocks, cfg.nkv,
                                     cfg.embedding_dim / cfg.nheads,
                                     GGML_TYPE_F16);
}

Inference::~Inference() = default;

ggml_context *Inference::init_context(uint64_t n_op_estimate) {
  uint64_t ctx_size = n_op_estimate * ggml_tensor_overhead() +
                      ggml_graph_overhead_custom(n_op_estimate, false) + 1024;
  if (ctx_buffer_.size() < ctx_size) {
    ctx_buffer_.resize(ctx_size);
  }
  ggml_init_params params = {.mem_size = ctx_size,
                             .mem_buffer = ctx_buffer_.data(),
                             .no_alloc = true};
  if (ctx_) {
    ggml_free(ctx_);
  }
  ctx_ = ggml_init(params);
  graph_ready_ = false;
  return ctx_;
}

ggml_cgraph *Inference::new_graph(uint32_t n_tokens, uint32_t max_context_len) {
  gf_ = ggml_new_graph(ctx_);
  model_.create_input_tensors(ctx_, n_tokens, max_context_len);
  return gf_;
}

void Inference::set_output(ggml_tensor *out) {
  output_ = out;
  ggml_build_forward_expand(gf_, output_);
}

bool Inference::prepare(uint32_t n_tokens) {
  ggml_backend_sched_reset(sched_.get());
  for (ggml_tensor *t : model_.get_input_tensors()) {
    ggml_backend_sched_set_tensor_backend(sched_.get(), t, model_.get_backend());
  }

  if (!ggml_backend_sched_reserve(sched_.get(), gf_)) {
    std::cerr << "Failed to reserve scheduler memory" << std::endl;
    return false;
  }
  ggml_backend_sched_alloc_graph(sched_.get(), gf_);

  cached_n_tokens_ = n_tokens;
  graph_ready_ = true;
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

  model_.set_inputs(input, past_tokens);

  auto status = ggml_backend_sched_graph_compute(sched_.get(), gf_);
  if (status != GGML_STATUS_SUCCESS) {
    std::cerr << "Graph compute failed with status: " << status << std::endl;
    return nullptr;
  }
  return output_;
}

std::vector<float> Inference::predict(std::vector<int32_t> tokens,
                                      int past_tokens) {
  model_config_t cfg = model_.config();
  const uint32_t n_tokens = (uint32_t)tokens.size();

  if (!graph_ready_ || n_tokens != cached_n_tokens_) {
    init_context(cfg.n_blocks * 64 + 256);
    new_graph(n_tokens, (uint32_t)cache_->max_len());
    ggml_tensor *output = model_.build_graph(ctx_, gf_, cache_.get());
    set_output(output);
    if (!prepare(n_tokens)) {
      return {};
    }
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

std::string Inference::invoke(std::string input, int max_tokens) {
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

  int i = 0;
  for (; i < max_tokens; i++) {
    if (past_tokens == 0) {
      events::prefill_start_event_t p_start;
      p_start.tokens_count = tokens.size();
      dispatcher.dispatch(InferenceEvent::PREFILL_STARTED, p_start);
    }
    std::vector<float> logits = predict(tokens, past_tokens);
    if (past_tokens == 0) {
      events::prefill_end_event_t p_end;
      p_end.tokens_count = tokens.size();
      dispatcher.dispatch(InferenceEvent::PREFILL_COMPLETED, p_end);
    }

    sampler_params_t params = {.temperature = 0.2f, .top_k = 0};
    Sampler sampler(params);
    std::pair<uint64_t, float> sample = sampler.sample(logits);
    past_tokens += tokens.size();
    tokens.clear();
    tokens.push_back(sample.first);
    std::string out_token =
        tokenizer.decode(static_cast<uint32_t>(sample.first));
    output += out_token;
    if (tokenizer.is_stop_token(out_token)) {
      dispatcher.dispatch(InferenceEvent::GENERATION_COMPLETED,
                          events::generation_completed_event_t{
                              .status = StreamStatus::EOS,
                              .num_tokens = static_cast<uint32_t>(i)});
      break;
    }

    dispatcher.dispatch(InferenceEvent::TOKEN_GENERATED,
                        events::token_generated_event_t{
                            .seq_id = static_cast<uint32_t>(i),
                            .token = std::string_view{out_token}});
  }
  if (i >= max_tokens) {
    dispatcher.dispatch(InferenceEvent::GENERATION_COMPLETED,
                        events::generation_completed_event_t{
                            .status = StreamStatus::MAX_CTX_REACHED});
  }
  return output;
}

} // namespace zinferlm
