#include <cstdint>
#include <cstring>
#include <ggml-cpp.h>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

#include "engine/graph.h"
#include "engine/tensors.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "layers/layers.h"
#include "model.h"
#include "model_loader/loader.h"
#include "sampling/samplers.h"

QwenModel::QwenModel(std::unique_ptr<ModelLoader> l) : loader_(std::move(l)) {}

zinferlm::model_info_t QwenModel::info() const { return loader_->info(); }

zinferlm::tokenizer_info_t QwenModel::tokenizer_info() const {
  return loader_->tokenizer_info();
}

std::vector<zinferlm::tensor_info_t> QwenModel::tensor_info() const {
  return loader_->tensor_info();
}

Layer *QwenModel::create_layer_(ggml_context *ctx, std::string layer_name,
                                Layers layer_type, int past_tokens, int len) {
  zinferlm::tensor_info_t info;
  Layer *layer = nullptr;
  switch (layer_type) {
  case Layers::TOKEN_EMBEDDING: {
    info = loader_->tensor_info(layer_name + ".weight");
    token_embedding_params_t params = {
        .ctx = ctx,
        .emb_w = create_tensor(
            ctx, info.name, static_cast<ggml_type>(info.type_id),
            loader_->get_tensor_ptr(info.data_offset), info.dimensions)};
    layer = new TokenEmbedding(params);
    break;
  }
  case Layers::TOKEN_UNEMBEDDING: {
    zinferlm::model_config_t config = loader_->model_config();
    zinferlm::tensor_info_t info = loader_->tensor_info(layer_name + ".weight");
    zinferlm::tensor_info_t norm_info =
        loader_->tensor_info(layer_name + "_norm.weight");
    token_unembedding_params_t uemb_params = {
        .ctx = ctx,
        .unemb_w = create_tensor(
            ctx, info.name, static_cast<ggml_type>(info.type_id),
            loader_->get_tensor_ptr(info.data_offset), info.dimensions),
        .norm_gamma = create_tensor(
            ctx, norm_info.name, static_cast<ggml_type>(norm_info.type_id),
            loader_->get_tensor_ptr(norm_info.data_offset),
            norm_info.dimensions),
        .norm_eps = config.rms_eps};
    layer = new TokenUnembedding(uemb_params);
    break;
  }
  case Layers::GROUPED_ATTN: {
    zinferlm::model_config_t config = loader_->model_config();
    zinferlm::tensor_info_t q_w =
        loader_->tensor_info(layer_name + "_q.weight");
    zinferlm::tensor_info_t q_b = loader_->tensor_info(layer_name + "_q.bias");
    zinferlm::tensor_info_t k_w =
        loader_->tensor_info(layer_name + "_k.weight");
    zinferlm::tensor_info_t k_b = loader_->tensor_info(layer_name + "_k.bias");
    zinferlm::tensor_info_t v_w =
        loader_->tensor_info(layer_name + "_v.weight");
    zinferlm::tensor_info_t v_b = loader_->tensor_info(layer_name + "_v.bias");
    zinferlm::tensor_info_t out_w =
        loader_->tensor_info(layer_name + "_output.weight");
    zinferlm::tensor_info_t norm_info =
        loader_->tensor_info(layer_name + "_norm.weight");
    grouped_attn_head_params attn_params = {
        .ctx = ctx,
        .q_w = create_tensor(ctx, q_w.name, static_cast<ggml_type>(q_w.type_id),
                             loader_->get_tensor_ptr(q_w.data_offset),
                             q_w.dimensions),
        .k_w = create_tensor(ctx, k_w.name, static_cast<ggml_type>(k_w.type_id),
                             loader_->get_tensor_ptr(k_w.data_offset),
                             k_w.dimensions),
        .q_b = create_tensor(ctx, q_b.name, static_cast<ggml_type>(q_b.type_id),
                             loader_->get_tensor_ptr(q_b.data_offset),
                             q_b.dimensions),
        .k_b = create_tensor(ctx, k_b.name, static_cast<ggml_type>(k_b.type_id),
                             loader_->get_tensor_ptr(k_b.data_offset),
                             k_b.dimensions),
        .v_w = create_tensor(ctx, v_w.name, static_cast<ggml_type>(v_w.type_id),
                             loader_->get_tensor_ptr(v_w.data_offset),
                             v_w.dimensions),
        .v_b = create_tensor(ctx, v_b.name, static_cast<ggml_type>(v_b.type_id),
                             loader_->get_tensor_ptr(v_b.data_offset),
                             v_b.dimensions),
        .out_w = create_tensor(
            ctx, out_w.name, static_cast<ggml_type>(out_w.type_id),
            loader_->get_tensor_ptr(out_w.data_offset), out_w.dimensions),
        .rope_freq_base = config.rope_freq_base,
        .apply_rope = true,
        .n_heads = config.nheads,
        .n_kv = config.nkv,
        .d_model = config.embedding_dim,
        .past_tokens = past_tokens,
        .len = len,
        .residual = true,
        .norm_gamma = create_tensor(
            ctx, norm_info.name, static_cast<ggml_type>(norm_info.type_id),
            loader_->get_tensor_ptr(norm_info.data_offset),
            norm_info.dimensions),
        .norm_eps = config.rms_eps,
        .debug = debug_,
        };
    layer = new GroupedAttentionHead(attn_params);
    break;
  }
  case Layers::SWIGLU: {
    zinferlm::model_config_t config = loader_->model_config();
    zinferlm::tensor_info_t up_w =
        loader_->tensor_info(layer_name + "_up.weight");
    zinferlm::tensor_info_t gate =
        loader_->tensor_info(layer_name + "_gate.weight");
    zinferlm::tensor_info_t down =
        loader_->tensor_info(layer_name + "_down.weight");
    zinferlm::tensor_info_t norm_info =
        loader_->tensor_info(layer_name + "_norm.weight");
    swiglu_params_t sparams = {
        .ctx = ctx,
        .w_gate = create_tensor(
            ctx, gate.name, static_cast<ggml_type>(gate.type_id),
            loader_->get_tensor_ptr(gate.data_offset), gate.dimensions),
        .w_down = create_tensor(
            ctx, down.name, static_cast<ggml_type>(down.type_id),
            loader_->get_tensor_ptr(down.data_offset), down.dimensions),
        .w_up = create_tensor(
            ctx, up_w.name, static_cast<ggml_type>(up_w.type_id),
            loader_->get_tensor_ptr(up_w.data_offset), up_w.dimensions),
        .residual = true,
        .norm_gamma = create_tensor(
            ctx, norm_info.name, static_cast<ggml_type>(norm_info.type_id),
            loader_->get_tensor_ptr(norm_info.data_offset),
            norm_info.dimensions),
        .norm_eps = config.rms_eps,
        };
    layer = new SwigLU(sparams);
    break;
  }
  }
  if (layer) {
    layer->name = layer_name;
  }
  return layer;
}

Block *QwenModel::create_attn_block_(ggml_context *ctx, int block_id,
                                     int past_tokens, int len) {
  std::vector<Layer *> layers;
  for (auto l : attn_blocks_) {
    layers.push_back(
        create_layer_(ctx, "blk." + std::to_string(block_id) + "." + l.second,
                      l.first, past_tokens, len));
  }
  block_params_t block_param = {
      .ctx = ctx,
      .layers = layers,
  };
  Block *block = new Block(block_param);
  block->name = "block_" + std::to_string(block_id);
  return block;
}

std::vector<std::unique_ptr<Layer>>
QwenModel::build_graph_(ggml_context *ctx, int past_tokens, int len) {
  uint32_t nblocks = loader_->model_config().n_blocks;
  std::vector<std::unique_ptr<Layer>> layers;
  int count = 0;
  zinferlm::tensor_info_t info;
  for (auto l : pre_attn_blocks_) {
    layers.push_back(std::unique_ptr<Layer>(
        create_layer_(ctx, l.second, l.first, past_tokens, len)));
  }
  for (uint32_t i = 0; i < nblocks; i++) {
    layers.push_back(
        std::unique_ptr<Layer>(create_attn_block_(ctx, i, past_tokens, len)));
  }
  for (auto l : post_attn_blocks_) {
    layers.push_back(std::unique_ptr<Layer>(
        create_layer_(ctx, l.second, l.first, past_tokens, len)));
  }
  return layers;
}

std::vector<float> QwenModel::invoke(std::vector<int32_t> tokens) {
  ggml_context_ptr ctx =
      init_engine(loader_->model_config().n_blocks * 64 + 256);
  std::vector<std::unique_ptr<Layer>> layers =
      build_graph_(ctx.get(), 0, tokens.size());
  std::vector<Layer *> layer_ptrs;
  layer_ptrs.reserve(layers.size());
  for (auto &l : layers) {
    layer_ptrs.push_back(l.get());
  }
  Graph graph(ctx.get(), layer_ptrs);
  graph.set_debug_mode(debug_);
  ggml_tensor *output = graph.execute(tokens);
  GGML_ASSERT(tokens.size() <= output->ne[1]);
  std::vector<float> logits(output->ne[0]);
  uint64_t offset =
      (tokens.size() - 1) * output->ne[0] * ggml_type_size(output->type);
  ggml_backend_tensor_get(output, logits.data(), offset,
                          output->ne[0] * ggml_type_size(output->type));
  return logits;
}

void QwenModel::summary() {
  ggml_context_ptr ctx = init_engine(loader_->get_tensor_count() * 48 + 128);
  std::vector<std::unique_ptr<Layer>> layers =
      build_graph_(ctx.get(), 0, 1);
  std::vector<Layer *> layer_ptrs;
  layer_ptrs.reserve(layers.size());
  for (auto &l : layers) {
    layer_ptrs.push_back(l.get());
  }
  Graph graph(ctx.get(), layer_ptrs);
  ggml_cgraph *gf = graph.build(1);
  ggml_graph_print(gf);
}
