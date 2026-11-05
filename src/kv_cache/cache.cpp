#include "cache.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "ggml-cpp.h"
#include "ggml.h"
#include "zinferlm/models.h"

namespace zinferlm {
namespace internal {

int KVCache::blocks_count() { return max_len_ / block_size_; }

void KVCache::init(int max_len, ggml_type type) {
  if (instance_) {
    return;
  }
  zinferlm::Model &model = zinferlm::Model::instance();
  zinferlm::model_config_t config = model.config();

  std::unique_ptr<KVCache> cache = std::make_unique<KVCache>(
      model.impl().get_backend(), max_len, config.n_blocks, config.nkv,
      config.embedding_dim / config.nheads, type);
  cache->block_tables_.reserve(config.n_blocks);

  size_t ctx_size = 2 * layers_ * ggml_tensor_overhead() + layers_ * 128;
  ggml_init_params params = {
      .mem_size = ctx_size,
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  ctx_kv_ = ggml_context_ptr{ggml_init(params)};

  for (int i = 0; i < config.n_blocks; i++) {
    cache->block_tables_[i].blocks_.reserve(blocks_count());
    cache->block_tables_[i].usage_.reserve(blocks_count());
    cache->block_tables_[i].entry.K = ggml_new_tensor_3d(
        ctx_kv_.get(), dtype_, d_head_ * n_kv_, block_size_, blocks_count());
    cache->block_tables_[i].entry.V = ggml_new_tensor_3d(
        ctx_kv_.get(), dtype_, d_head_ * n_kv_, block_size_, blocks_count());
    ggml_set_name(cache->block_tables_[i].entry.K,
                  ("cache->k_l" + std::to_string(i)).c_str());
    ggml_set_name(cache->block_tables_[i].entry.V,
                  ("cache->v_l" + std::to_string(i)).c_str());
  }

  buffer_ = ggml_backend_buffer_ptr{ggml_backend_alloc_ctx_tensors_from_buft(
      ctx_kv_.get(),
      ggml_backend_get_default_buffer_type(model.impl().get_backend()))};

  instance_ = std::move(cache);
}

void KVCache::set_graph(ggml_cgraph *gf) { gf_ = gf; }

void KVCache::reset() {}
//
// void KVCache::write_kv(ggml_context *ctx, int l, ggml_tensor *K, ggml_tensor
// *V,
//                        ggml_tensor *pos_idx) {
//   layer_kv_cache_t &cache = layers_[l];
//   const uint64_t seq_len = K->ne[2];
//   const uint64_t n_embed = d_head_ * n_kv_;
//
//   ggml_tensor *k_cur = ggml_view_2d(ctx, K, n_embed, seq_len,
//                                     ggml_row_size(K->type, n_embed), 0);
//   ggml_tensor *v_cur = ggml_view_2d(ctx, V, n_embed, seq_len,
//                                     ggml_row_size(V->type, n_embed), 0);
//
//   ggml_tensor *k_set = ggml_set_rows(ctx, cache.K, k_cur, pos_idx);
//   ggml_tensor *v_set = ggml_set_rows(ctx, cache.V, v_cur, pos_idx);
//
//   ggml_set_name(k_set, ("set_rows_k_l" + std::to_string(l)).c_str());
//   ggml_set_name(v_set, ("set_rows_v_l" + std::to_string(l)).c_str());
//
//   ggml_build_forward_expand(gf_, k_set);
//   ggml_build_forward_expand(gf_, v_set);
// }
//

int KVCache::alloc_block(int layer) {
  std::lock_guard<std::mutex> lock(mutex_);
  kv_cache_table_t &table = block_tables_[layer];
  for (int i = 0; i < table.usage_.size(); i++) {
    if (!table.usage_[i]) {
      table.usage_[i] = true;
      table.blocks_[i].num_entries = 0;
      table.blocks_[i].seq_id = -1;
      table.blocks_[i].start_pos = 0;
      return i;
    }
  }
  return -1;
}

void KVCache::write_kv(ggml_context *ctx, int l, ggml_tensor *K,
                       ggml_tensor *V) {
  const uint64_t emb_dim = d_head_ * n_kv_;
  const uint64_t seq_len = K->ne[2];
  int seq_id = 0;
  kv_cache_table_t &table = block_tables_[l];
  kv_cache_usage_t &usage = usage_table_[seq_id];
  bool is_appended = false;
  int tokens_stored = 0;
  while (tokens_stored < seq_len) {
    int block_id = -1;
    if (usage.block_list.size() > 0) {
      kv_cache_block_t last_block = table.blocks_[usage.block_list.back()];
      if (last_block.num_entries < block_size_) {
        block_id = usage.block_list.back();
      }
    }
    if (block_id == -1) {
      block_id = alloc_block(l);
    }
    kv_cache_block_t &block = table.blocks_[block_id];
    block.start_pos = usage.seq_len;
    block.seq_id = seq_id;
    int remaining = seq_len - tokens_stored;
    int len = std::min(block_size_ - block.num_entries, remaining);
    ggml_tensor *K_cached =
        ggml_view_2d(ctx, table.entry.K, emb_dim, len, table.entry.K->nb[1],
                     block_id * table.entry.K->nb[2] +
                         block.num_entries * table.entry.K->nb[1]);
    ggml_tensor *K_cur =
        ggml_view_2d(ctx, K, emb_dim, len, ggml_row_size(K->type, emb_dim),
                     seq_id * K->nb[3] + tokens_stored * K->nb[2]);

    ggml_tensor *V_cached =
        ggml_view_2d(ctx, table.entry.V, emb_dim, len, table.entry.V->nb[1],
                     block_id * table.entry.V->nb[2] +
                         block.num_entries * table.entry.V->nb[1]);
    ggml_tensor *V_cur =
        ggml_view_2d(ctx, V, emb_dim, len, ggml_row_size(V->type, emb_dim),
                     seq_id * V->nb[3] + tokens_stored * V->nb[2]);

    ggml_tensor *K_slice = ggml_cpy(ctx, K_cur, K_cached);
    ggml_tensor *V_slice = ggml_cpy(ctx, V_cur, V_cached);
    ggml_build_forward_expand(gf_, K_slice);
    ggml_build_forward_expand(gf_, V_slice);
    tokens_stored += len;
    block.num_entries += len;
    usage_table_[seq_id].block_list.push_back(block_id);
  }
  usage_table_[seq_id].seq_len += tokens_stored;
}
//
// layer_kv_cache_t KVCache::get_slice(ggml_context *graph_ctx, int l) {
//   layer_kv_cache_t cache = layers_[l];
//   const uint64_t n_emb = d_head_ * n_kv_;
//   return {
//       .K = ggml_view_4d(graph_ctx, cache.K, d_head_, max_len_, n_kv_, 1,
//                         ggml_row_size(cache.K->type, n_emb),
//                         ggml_row_size(cache.K->type, d_head_),
//                         ggml_row_size(cache.K->type, n_emb * max_len_), 0),
//       .V = ggml_view_4d(graph_ctx, cache.V, d_head_, max_len_, n_kv_, 1,
//                         ggml_row_size(cache.V->type, n_emb),
//                         ggml_row_size(cache.V->type, d_head_),
//                         ggml_row_size(cache.V->type, n_emb * max_len_), 0),
//
//   };
// }

std::pair<ggml_tensor*, ggml_tensor*> KVCache::get_slice(ggml_context* graph_ctx, int l) {
  kv_cache_table_t table = block_tables_[l];
  int seq_id = 0;
  kv_cache_usage_t usage = usage_table_[seq_id];
  for (auto& block: usage.block_list) {
    
  }
}

} // namespace internal
} // namespace zinferlm
