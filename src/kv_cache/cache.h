#pragma once

#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include <memory>
#include <vector>

struct layer_kv_cache_t {
  ggml_tensor* K; // [d_head * n_kv, seq_len] head-major
  ggml_tensor* V;
};

class KVCache {
  private:
    int max_len_;
    int d_head_;
    int n_kv_;
    ggml_cgraph *gf_;
    std::vector<layer_kv_cache_t> layers_;
    ggml_context_ptr ctx_kv_;
    ggml_backend_buffer_ptr buffer_;

  public:
    KVCache(ggml_backend *backend, int ctx_len, int layers, int n_kv,
            int d_head, ggml_type type);
    void set_graph(ggml_cgraph *gf);
    void reset();
    void write_kv(ggml_context* ctx, int l, ggml_tensor* K, ggml_tensor* V, ggml_tensor* pos_idx);
    layer_kv_cache_t get_slice(ggml_context* graph_ctx, int layer);
    int max_len() const { return max_len_; }
};
