#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml.h"

namespace zinferlm {
namespace internal {

struct kv_cache_data_t {
  std::vector<ggml_tensor *> K;
  std::vector<ggml_tensor *> V;
};

struct cache_entry_t {
  ggml_tensor *K;
  ggml_tensor *V;
};

struct kv_cache_block_t {
  int seq_id;
  int start_pos;
  int num_entries;
};

struct kv_cache_table_t {
  std::vector<kv_cache_block_t> blocks_;
  std::vector<bool> usage_;
  cache_entry_t entry;
};

struct kv_cache_usage_t {
  int seq_id;
  std::vector<int> block_list;
  int seq_len;
};

class KVCache {
private:
  std::unique_ptr<KVCache> instance_;

  std::mutex mutex_;

  int max_len_;
  int d_head_;
  int n_kv_;
  int layers_;
  int block_size_;
  ggml_type dtype_;
  ggml_cgraph *gf_;
  std::vector<kv_cache_table_t> block_tables_;
  std::unordered_map<int, kv_cache_usage_t> usage_table_;
  ggml_context_ptr ctx_kv_;
  ggml_backend_buffer_ptr buffer_;
  KVCache(ggml_backend *backend, int ctx_len, int layers, int n_kv, int d_head,
          ggml_type type = GGML_TYPE_F16)
      : max_len_(ctx_len), layers_(layers), d_head_(d_head), n_kv_(n_kv),
        dtype_(type) {
    block_size_ = 16;
  }
  int alloc_block(int layer);

public:
  KVCache(const KVCache &c) = delete;
  KVCache &operator=(const KVCache &) = delete;
  void init(int max_len, ggml_type type);
  void set_graph(ggml_cgraph *gf);
  void reset();
  void write_kv(ggml_context *ctx, int l, ggml_tensor *K, ggml_tensor *V);
  std::pair<ggml_tensor*, ggml_tensor*> get_slice(ggml_context *graph_ctx, int layer);
  int blocks_count();
};

} // namespace internal
} // namespace zinferlm
