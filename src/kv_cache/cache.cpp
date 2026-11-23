#include "cache.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "engine/measure.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "logging/log.h"
#include "zinferlm/models.h"

namespace zinferlm {
namespace internal {

size_t KVCache::blocks_count() const { return 4 * max_len_ / block_size_; }

KVCache::KVCache(ggml_backend *backend, int ctx_len, int layers, int n_kv,
                 int d_head, ggml_type type, bool measure_only)
    : max_len_(ctx_len), layers_(layers), d_head_(d_head), n_kv_(n_kv),
      dtype_(type), blocks_used_(0) {
  block_size_ = 16;
  size_per_block_ =
      2ull * layers_ * d_head_ * n_kv_ * block_size_ * ggml_type_size(type);
  LOG_INF("Creating kv cache layers=%d, block_size=%d, total_blocks=%zu, "
          "size_per_block=%zu, reserved_size=%zu, measure_only=%d\n",
          layers, block_size_, blocks_count(), size_per_block_,
          size_per_block_ * blocks_count(), (int)measure_only);

  free_.resize(blocks_count(), false);
  blocks_.resize(blocks_count());
  pool_.resize(layers_);

  size_t ctx_size = 2 * layers_ * ggml_tensor_overhead() + layers_ * 128 + 2;
  ggml_init_params params = {
      .mem_size = ctx_size,
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  ctx_kv_ = ggml_context_ptr{ggml_init(params)};

  for (int i = 0; i < layers_; i++) {
    pool_[i].entry.K = ggml_new_tensor_3d(
        ctx_kv_.get(), dtype_, d_head_ * n_kv_, block_size_, blocks_count());
    pool_[i].entry.V = ggml_new_tensor_3d(
        ctx_kv_.get(), dtype_, d_head_ * n_kv_, block_size_, blocks_count());
    ggml_set_name(pool_[i].entry.K, ("cache->k_l" + std::to_string(i)).c_str());
    ggml_set_name(pool_[i].entry.V, ("cache->v_l" + std::to_string(i)).c_str());
  }

  if (measure_only) {
    // No device allocation: the pool owns its own buffer in a real run, so the
    // compute-buffer planner must not count it.
    for (int i = 0; i < layers_; i++) {
      mark_external_tensor(pool_[i].entry.K);
      mark_external_tensor(pool_[i].entry.V);
    }
  } else {
    buffer_ = ggml_backend_buffer_ptr{ggml_backend_alloc_ctx_tensors_from_buft(
        ctx_kv_.get(), ggml_backend_get_default_buffer_type(backend))};
  }
}

std::vector<ggml_tensor *> KVCache::tensors() const {
  std::vector<ggml_tensor *> out;
  out.reserve(pool_.size() * 2);
  for (const auto &entry : pool_) {
    out.push_back(entry.entry.K);
    out.push_back(entry.entry.V);
  }
  return out;
}

void KVCache::set_graph(ggml_cgraph *gf) { gf_ = gf; }

void KVCache::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::fill(free_.begin(), free_.end(), false);
  for (auto &b : blocks_) {
    b = {.slot_id = -1, .start_pos = 0, .num_entries = 0};
  }
  usage_table_.clear();
}

int KVCache::alloc_block() {
  for (int i = 0; i < (int)free_.size(); i++) {
    if (!free_[i]) {
      free_[i] = true;
      blocks_[i] = {.slot_id = -1, .start_pos = 0, .num_entries = 0};
      blocks_used_ += 1;
      return i;
    }
  }
  return -1;
}

void KVCache::ensure_blocks(int slot_id, size_t filled) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto &&it = usage_table_.find(slot_id);
  if (it == usage_table_.end()) {
    usage_table_[slot_id] = {.slot_id = slot_id, .seq_len = 0};
  }
  kv_cache_usage_t &usage = usage_table_[slot_id];
  size_t needed = (filled + block_size_ - 1) / block_size_;
  while (usage.block_list.size() < needed) {
    int b = alloc_block();
    if (b < 0) {
      LOG_WRN("ensure_blocks: OUT OF BLOCKS! seq=%d filled=%zu needed=%zu "
              "allocated=%zu\n",
              slot_id, filled, needed, usage.block_list.size());
      break;
    }
    usage.block_list.push_back(b);
  }
  LOG_DBG("ensure_blocks: seq=%d filled=%zu needed=%zu allocated=%zu "
          "free_count=%zu\n",
          slot_id, filled, needed, usage.block_list.size(),
          blocks_count() - blocks_used_);
}

void KVCache::free_blocks(int slot_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = usage_table_.find(slot_id);
  if (it == usage_table_.end()) {
    return;
  }
  for (int b : it->second.block_list) {
    free_[b] = false;
    blocks_[b] = {.slot_id = -1, .start_pos = 0, .num_entries = 0};
    blocks_used_--;
  }
  usage_table_.erase(it);
}

std::vector<int32_t> KVCache::get_lookup_rows(int slot_id,
                                              size_t filled) const {
  const auto &blocks = usage_table_.at(slot_id).block_list;
  std::vector<int32_t> rows(filled);
  for (size_t p = 0; p < filled; p++) {
    rows[p] = blocks[p / block_size_] * block_size_ +
              static_cast<int>(p % block_size_);
  }
  return rows;
}

std::vector<int64_t> KVCache::get_write_rows(int slot_id, size_t past_tokens,
                                             size_t n_tokens) const {
  const auto &blocks = usage_table_.at(slot_id).block_list;
  std::vector<int64_t> rows(n_tokens);
  LOG_DBG("get_write_rows: block_list.size=%zu blocks_count=%zu past_tokens=%zu "
          "n_tokens=%zu\n",
          blocks.size(), blocks_count(), past_tokens, n_tokens);
  for (size_t i = 0; i < n_tokens; i++) {
    size_t pos = past_tokens + i;
    size_t logical = pos / block_size_;
    if (logical >= blocks.size()) {
      LOG_DBG("get_write_rows: OOB logical=%zu >= block_list.size=%zu\n",
              logical, blocks.size());
      break;
    }
    rows[i] =
        blocks[logical] * block_size_ + static_cast<int>(pos % block_size_);
  }
  return rows;
}

void KVCache::write_kv(ggml_context *ctx, int layer, ggml_tensor *K,
                       ggml_tensor *V, ggml_tensor *write_rows) {
  const int emb_dim = d_head_ * n_kv_;
  const int n_tokens = K->ne[2];
  cache_entry_t &entry = pool_[layer].entry;

  ggml_tensor *K_cached = ggml_view_2d(
      ctx, entry.K, emb_dim, block_size_ * blocks_count(), entry.K->nb[1], 0);
  ggml_tensor *V_cached = ggml_view_2d(
      ctx, entry.V, emb_dim, block_size_ * blocks_count(), entry.V->nb[1], 0);

  ggml_tensor *K_cur = ggml_reshape_2d(ctx, K, emb_dim, n_tokens);
  ggml_tensor *V_cur = ggml_reshape_2d(ctx, V, emb_dim, n_tokens);

  ggml_build_forward_expand(gf_,
                            ggml_set_rows(ctx, K_cached, K_cur, write_rows));
  ggml_build_forward_expand(gf_,
                            ggml_set_rows(ctx, V_cached, V_cur, write_rows));
}

std::pair<ggml_tensor *, ggml_tensor *>
KVCache::get_slice(ggml_context *graph_ctx, int l, ggml_tensor *lookup_rows) {
  cache_entry_t &entry = pool_[l].entry;
  ggml_tensor *K_flat =
      ggml_view_2d(graph_ctx, entry.K, entry.K->ne[0],
                   block_size_ * blocks_count(), entry.K->nb[1], 0);
  ggml_tensor *V_flat =
      ggml_view_2d(graph_ctx, entry.V, entry.V->ne[0],
                   block_size_ * blocks_count(), entry.V->nb[1], 0);
  ggml_tensor *K_slice = ggml_get_rows(graph_ctx, K_flat, lookup_rows);
  ggml_tensor *V_slice = ggml_get_rows(graph_ctx, V_flat, lookup_rows);

  K_slice = ggml_view_3d(graph_ctx, K_slice, d_head_, lookup_rows->ne[0], n_kv_,
                         ggml_row_size(K_slice->type, d_head_ * n_kv_),
                         ggml_row_size(K_slice->type, d_head_), 0);
  V_slice = ggml_view_3d(graph_ctx, V_slice, d_head_, lookup_rows->ne[0], n_kv_,
                         ggml_row_size(V_slice->type, d_head_ * n_kv_),
                         ggml_row_size(V_slice->type, d_head_), 0);
  return std::make_pair(K_slice, V_slice);
}

} // namespace internal
} // namespace zinferlm
