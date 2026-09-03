#include <cstdint>
#include <ggml-cpp.h>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

#include "engine/graph.h"
#include "engine/tensors.h"
#include "ggml.h"
#include "layers/layers.h"
#include "model.h"

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
  switch (layer_type) {
  case Layers::TOKEN_EMBEDDING: {
    info = loader_->tensor_info(layer_name + ".weight");
    token_embedding_params_t params = {
        .ctx = ctx,
        .emb_w = create_tensor(
            ctx, info.name, static_cast<ggml_type>(info.type_id),
            loader_->get_tensor_ptr(info.data_offset), info.dimensions)};
    return new TokenEmbedding(params);
  }
  case Layers::NORM: {
    zinferlm::tensor_info_t info = loader_->tensor_info(layer_name + ".weight");
    norm_params_t p = {.ctx = ctx,
                       .gamma = create_tensor(
                           ctx, info.name, static_cast<ggml_type>(info.type_id),
                           loader_->get_tensor_ptr(info.data_offset),
                           info.dimensions),
                       .eps = 10e-8};
    return new NormLayer(p);
  }
  case Layers::TOKEN_UNEMBEDDING: {
    zinferlm::tensor_info_t info = loader_->tensor_info(layer_name + ".weight");
    token_unembedding_params_t uemb_params = {
        .ctx = ctx,
        .unemb_w = create_tensor(
            ctx, info.name, static_cast<ggml_type>(info.type_id),
            loader_->get_tensor_ptr(info.data_offset), info.dimensions)};
    return new TokenUnembedding(uemb_params);
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
        .residual = true};
    return new GroupedAttentionHead(attn_params);
  }
  case Layers::SWIGLU: {
    zinferlm::tensor_info_t up_w =
        loader_->tensor_info(layer_name + "_up.weight");
    zinferlm::tensor_info_t gate =
        loader_->tensor_info(layer_name + "_gate.weight");
    zinferlm::tensor_info_t down =
        loader_->tensor_info(layer_name + "_down.weight");
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
        .residual = true};
    return new SwigLU(sparams);
  }
  }
  return nullptr;
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
  return new Block(block_param);
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
  for (int i = 0; i < nblocks; i++) {
    layers.push_back(
        std::unique_ptr<Layer>(create_attn_block_(ctx, i, past_tokens, len)));
  }
  for (auto l : post_attn_blocks_) {
    layers.push_back(std::unique_ptr<Layer>(
        create_layer_(ctx, l.second, l.first, past_tokens, len)));
  }
  // for (auto t : loader_->tensor_info()) {
  //   if (t.name == "output.weight") {
  //     token_unembedding_params_t params = {
  //         .ctx = ctx,
  //         .unemb_w = create_tensor(ctx, static_cast<ggml_type>(t.type_id),
  //                                  loader_->get_tensor_ptr(t.data_offset),
  //                                  t.dimensions)};
  //     layers.emplace_back(std::make_unique<TokenUnembedding>(params));
  //   }
  // }
  return layers;
}

std::vector<uint32_t> QwenModel::tokenize(std::string input) {
  zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(*this);
  std::vector<std::pair<std::string, uint64_t>> tokens =
      tokenizer.tokenize(input);
  std::vector<uint32_t> token_ids;
  for (auto &p : tokens) {
    token_ids.push_back(p.second);
  }
  return token_ids;
}

std::string QwenModel::invoke(std::string input) {
  std::vector<uint32_t> token_ids = tokenize(input);
  ggml_context_ptr ctx = init_engine(loader_->get_tensor_count());
  std::vector<std::unique_ptr<Layer>> layers =
      build_graph_(ctx.get(), 0, token_ids.size());
  std::vector<Layer *> layer_ptrs;
  layer_ptrs.reserve(layers.size());
  for (auto &l : layers) {
    layer_ptrs.push_back(l.get());
  }
  Graph graph(ctx.get(), layer_ptrs);
  ggml_tensor *output = graph.execute(token_ids);
  std::cout << output->ne[1] << " " << output->ne[0] << std::endl;
  return "";
}

void QwenModel::summary(std::string input) {
  std::vector<uint32_t> token_ids = tokenize(input);
  ggml_context_ptr ctx = init_engine(loader_->get_tensor_count());
  std::vector<std::unique_ptr<Layer>> layers =
      build_graph_(ctx.get(), 0, token_ids.size());
  std::vector<Layer *> layer_ptrs;
  layer_ptrs.reserve(layers.size());
  for (auto &l : layers) {
    layer_ptrs.push_back(l.get());
  }
  std::cout << "layers completed" << std::endl;
  Graph graph(ctx.get(), layer_ptrs);
  ggml_cgraph *gf = graph.build(token_ids.size());
  ggml_graph_print(gf);
}
