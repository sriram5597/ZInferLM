#include "cache.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "zinferlm/models.h"
#include <cstdint>
#include <string>

KVCache::KVCache(ggml_backend *backend, int ctx_len, int layers, int n_kv,
                 int d_head, ggml_type type = GGML_TYPE_F16)
    : max_len_(ctx_len), layers_(layers), d_head_(d_head), n_kv_(n_kv) {
  size_t ctx_size = 2 * layers * ggml_tensor_overhead() + layers * 128;
  ggml_init_params params = {
      .mem_size = ctx_size,
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  ctx_kv_ = ggml_context_ptr{ggml_init(params)};
  for (int i = 0; i < layers; i++) {
    layers_[i] = {
        .K = ggml_new_tensor_2d(ctx_kv_.get(), type, d_head * n_kv_, ctx_len),
        .V = ggml_new_tensor_2d(ctx_kv_.get(), type, d_head * n_kv_, ctx_len)};
    ggml_set_name(layers_[i].K, ("cache_k_l" + std::to_string(i)).c_str());
    ggml_set_name(layers_[i].V, ("cache_v_l" + std::to_string(i)).c_str());
  }
  buffer_ = ggml_backend_buffer_ptr{ggml_backend_alloc_ctx_tensors_from_buft(
      ctx_kv_.get(), ggml_backend_get_default_buffer_type(backend))};
}

void KVCache::set_graph(ggml_cgraph *gf) { gf_ = gf; }

void KVCache::reset() {}

void KVCache::write_kv(ggml_context *ctx, int l, ggml_tensor *K, ggml_tensor *V,
                       ggml_tensor *pos_idx) {
  layer_kv_cache_t &cache = layers_[l];
  const uint64_t seq_len = K->ne[2];
  const uint64_t n_embed = d_head_ * n_kv_;

  ggml_tensor *k_cur = ggml_view_2d(ctx, K, n_embed, seq_len,
                                    ggml_row_size(K->type, n_embed), 0);
  ggml_tensor *v_cur = ggml_view_2d(ctx, V, n_embed, seq_len,
                                    ggml_row_size(V->type, n_embed), 0);

  ggml_tensor *k_set = ggml_set_rows(ctx, cache.K, k_cur, pos_idx);
  ggml_tensor *v_set = ggml_set_rows(ctx, cache.V, v_cur, pos_idx);

  ggml_set_name(k_set, ("set_rows_k_l" + std::to_string(l)).c_str());
  ggml_set_name(v_set, ("set_rows_v_l" + std::to_string(l)).c_str());

  ggml_build_forward_expand(gf_, k_set);
  ggml_build_forward_expand(gf_, v_set);
}

layer_kv_cache_t KVCache::get_slice(ggml_context *graph_ctx, int l) {
  layer_kv_cache_t cache = layers_[l];
  const uint64_t n_emb = d_head_ * n_kv_;
  return {
      .K = ggml_view_4d(graph_ctx, cache.K, d_head_, max_len_, n_kv_, 1,
                        ggml_row_size(cache.K->type, n_emb),
                        ggml_row_size(cache.K->type, d_head_),
                        ggml_row_size(cache.K->type, n_emb * max_len_), 0),
      .V = ggml_view_4d(graph_ctx, cache.V, d_head_, max_len_, n_kv_, 1,
                        ggml_row_size(cache.V->type, n_emb),
                        ggml_row_size(cache.V->type, d_head_),
                        ggml_row_size(cache.V->type, n_emb * max_len_), 0),

  };
}
