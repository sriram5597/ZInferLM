#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml.h"

namespace zinferlm {
namespace internal {

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
  cache_entry_t entry;
};

struct kv_cache_usage_t {
  int seq_id;
  std::vector<int> block_list;
  int seq_len;
};

class KVCache {
private:
  std::mutex mutex_;

  int max_len_;
  int d_head_;
  int n_kv_;
  int layers_;
  int block_size_;
  int blocks_used_;
  ggml_type dtype_;
  ggml_cgraph *gf_;

  std::vector<bool> free_;
  std::vector<kv_cache_block_t> blocks_;
  std::vector<kv_cache_table_t> pool_;
  std::unordered_map<int, kv_cache_usage_t> usage_table_;

  ggml_context_ptr ctx_kv_;
  ggml_backend_buffer_ptr buffer_;

  int alloc_block();

public:
  KVCache(ggml_backend *backend, int ctx_len, int layers, int n_kv, int d_head,
          ggml_type type = GGML_TYPE_F16);

  KVCache(const KVCache &c) = delete;
  KVCache &operator=(const KVCache &) = delete;

  void set_graph(ggml_cgraph *gf);
  void reset();
  void ensure_blocks(int seq_id, size_t filled);
  void free_blocks(int seq_id);
  std::vector<int32_t> get_lookup_rows(int seq_id, size_t filled) const;
  std::vector<int64_t> get_write_rows(int seq_id, size_t past_tokens,
                                       size_t n_tokens) const;
    void write_kv(ggml_context * ctx, int layer, ggml_tensor *K,
                  ggml_tensor *V, ggml_tensor *write_rows);
    std::pair<ggml_tensor *, ggml_tensor *> get_slice(
        ggml_context * graph_ctx, int layer, ggml_tensor *lookup_rows);
    int blocks_count() const;
  };
} // namespace internal
} // namespace zinferlm
