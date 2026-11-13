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
  cache_->ensure_blocks(0, max_ctx + prefill_chunksize_);
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

Inference::~Inference() { ggml_free(ctx_); }

ggml_cgraph *Inference::new_graph(size_t n_tokens, size_t total_filled) {
  gf_ = ggml_new_graph(ctx_);
  model_impl_.create_input_tensors(ctx_, n_tokens, total_filled);
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

void Inference::compute(const model_input_t &input) {
  if (debug_) {
    debug_cb_data_ = {debug_, {}};
    ggml_backend_sched_set_eval_callback(sched_.get(), debug_eval_callback,
                                         &debug_cb_data_);
  } else {
    ggml_backend_sched_set_eval_callback(sched_.get(), nullptr, nullptr);
  }

  if (debug_) {
    std::cout << "Input Tokens: [ ";
    for (auto t : input.tokens) {
      std::cout << t << ", ";
    }
    std::cout << " ]\n";
  }

  auto status = ggml_backend_sched_graph_compute(sched_.get(), gf_);
  if (status != GGML_STATUS_SUCCESS) {
    std::cerr << "Graph compute failed with status: " << status << std::endl;
  }
}

std::pair<std::vector<float>, int64_t>
Inference::predict(const model_input_t &input, size_t n_tokens,
                   size_t total_filled) {
  zinferlm::model_config_t cfg = model_impl_.config();

  init_context(cfg.n_blocks * 64 + 256);
  new_graph(n_tokens, total_filled);
  ggml_tensor *output = model_impl_.build_graph(ctx_, gf_, cache_);
  ggml_tensor *out_rows =
      ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, input.offsets.size());
  ggml_set_input(out_rows);
  ggml_backend_sched_set_tensor_backend(sched_.get(), out_rows,
                                        model_impl_.get_backend());
  ggml_tensor *aggregated_out = ggml_get_rows(ctx_, output, out_rows);
  set_output(aggregated_out);
  if (!prepare()) {
    return {};
  }
  model_impl_.set_inputs(input);
  ggml_backend_tensor_set(out_rows, input.offsets.data(), 0,
                          out_rows->ne[0] * ggml_type_size(GGML_TYPE_I32));

  compute(input);
  std::vector<float> logits(input.offsets.size() * aggregated_out->ne[0]);
  ggml_backend_tensor_get(aggregated_out, logits.data(), 0,
                          aggregated_out->ne[0] * aggregated_out->ne[1] *
                              ggml_type_size(aggregated_out->type));
  return std::make_pair(logits, aggregated_out->ne[0]);
}
} // namespace internal
} // namespace zinferlm
