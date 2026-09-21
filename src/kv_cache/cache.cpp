#include "cache.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include <string>

KVCache::KVCache(ggml_backend_buffer_type_t buf_type, int ctx_len, int layers,
                 int n_kv, int d_head, ggml_type type = GGML_TYPE_F16)
    : layers_(layers), d_head_(d_head), n_kv_(n_kv) {
  size_t ctx_size = 2 * layers * ggml_tensor_overhead() + 4096;
  ggml_init_params params = {
      .mem_size = ctx_size,
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  ctx_kv_ = ggml_context_ptr{ggml_init(params)};
  for (int i = 0; i < layers; i++) {
    layers_[i] = {
        .len = 0,
        .K = ggml_new_tensor_4d(ctx_kv_.get(), type, d_head, ctx_len, n_kv, 1),
        .V = ggml_new_tensor_4d(ctx_kv_.get(), type, d_head, ctx_len, n_kv, 1)};
    ggml_set_name(layers_[i].K, ("cache_k_l" + std::to_string(i)).c_str());
    ggml_set_name(layers_[i].V, ("cache_v_l" + std::to_string(i)).c_str());
  }
  buffer_ = ggml_backend_buffer_ptr{
      ggml_backend_alloc_ctx_tensors_from_buft(ctx_kv_.get(), buf_type)};
}

void KVCache::set_graph(ggml_cgraph *gf) { gf_ = gf; }

void KVCache::reset() {
  for (auto& layer : layers_) {
    layer.len = 0;
  }
}

void KVCache::concat(ggml_context* ctx, int l, int seq_len, ggml_tensor *K, ggml_tensor *V) {
  layer_kv_cache_t &cache = layers_[l];
  size_t offset = cache.len * cache.K->nb[1];
  ggml_tensor *k_view =
      ggml_view_4d(ctx, cache.K, d_head_, seq_len, n_kv_, 1,
                   cache.K->nb[1], cache.K->nb[2], cache.K->nb[3], offset);
  ggml_tensor *v_view =
      ggml_view_4d(ctx, cache.V, d_head_, seq_len, n_kv_, 1,
                   cache.V->nb[1], cache.V->nb[2], cache.V->nb[3], offset);
  ggml_tensor *k_cpy = ggml_cpy(ctx, K, k_view);
  ggml_tensor *v_cpy = ggml_cpy(ctx, V, v_view);
  ggml_set_name(k_cpy, ("cpy_k_l" + std::to_string(l)).c_str());
  ggml_set_name(v_cpy, ("cpy_v_l" + std::to_string(l)).c_str());
  ggml_build_forward_expand(gf_, k_cpy);
  ggml_build_forward_expand(gf_, v_cpy);
  cache.len += seq_len;
}

layer_kv_cache_t KVCache::get_slice(ggml_context* graph_ctx, int l) {
  layer_kv_cache_t cache = layers_[l];
  return {
      .K = ggml_view_4d(graph_ctx, cache.K, d_head_, cache.len, n_kv_, 1,
                        cache.K->nb[1], cache.K->nb[2], cache.K->nb[3], 0),
      .V = ggml_view_4d(graph_ctx, cache.V, d_head_, cache.len, n_kv_, 1,
                        cache.V->nb[1], cache.V->nb[2], cache.V->nb[3], 0),
  };
}
