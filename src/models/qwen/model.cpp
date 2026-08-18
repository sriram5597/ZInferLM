#include <ggml-cpp.h>
#include <memory>
#include <iostream>
#include <memory>
#include <ostream>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

#include "engine/graph.h"
#include "engine/tensors.h"
#include "layers/layers.h"
#include "model.h"

zinferlm::model_info_t QwenModel::info() const
{
    return loader_->info();
}

zinferlm::tokenizer_info_t QwenModel::tokenizer_info() const
{
    return loader_->tokenizer_info();
}

std::vector<zinferlm::tensor_info_t> QwenModel::tensor_info() const
{
    return loader_->tensor_info();
}

std::vector<std::unique_ptr<Layer>> QwenModel::build_graph_(ggml_context *ctx)
{
    std::vector<std::unique_ptr<Layer>> layers;
    int count = 0;
    for (auto t : loader_->tensor_info())
    {
        if (t.name == "token_embd.weight")
        {
            token_embedding_params_t params = {
              .ctx=ctx_.get(),
              .emb_w=create_tensor(ctx_.get(), static_cast<ggml_type>(t.type_id), loader_->get_tensor_ptr(t.data_offset), t.dimensions)
            };
            layers.emplace(layers.begin(), std::make_unique<TokenEmbedding>(params));
        }
        if (t.name == "output.weight")
        {
          token_unembedding_params_t params = {
              .ctx=ctx_.get(),
              .unemb_w=create_tensor(ctx_.get(), static_cast<ggml_type>(t.type_id), loader_->get_tensor_ptr(t.data_offset), t.dimensions)
            };
            layers.emplace_back(std::make_unique<TokenUnembedding>(params));
        }
        // std::cout << t.name << " -- " << ggml_type_name(static_cast<ggml_type>(t.type_id)) << std::endl;  
    }
    for (auto &l : layers)
    {
        std::cout << l->name << " ";
    }
    std::cout << std::endl;
    return layers;
}

std::vector<uint32_t> QwenModel::tokenize(std::string input)
{
    zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(*this);
    std::vector<std::pair<std::string, uint64_t>> tokens = tokenizer.tokenize(input);
    std::vector<uint32_t> token_ids;
    for (auto &p : tokens)
    {
        token_ids.push_back(p.second);
    }
    return token_ids;
}

std::string QwenModel::invoke(std::string input)
{
    std::vector<uint32_t> token_ids = tokenize(input);
    ggml_context_ptr ctx = init_engine(loader_->get_tensor_count());
    std::vector<std::unique_ptr<Layer>> layers = build_graph_(ctx.get());
    std::vector<Layer *> layer_ptrs;
    layer_ptrs.reserve(layers.size());
    for (auto &l : layers)
    {
        layer_ptrs.push_back(l.get());
    }
    Graph graph(ctx.get(), layer_ptrs);
    ggml_tensor *output = graph.execute(token_ids);
    std::cout << output->ne[1] << " " << output->ne[0] << std::endl;
    return "";
}

void QwenModel::summary(std::string input)
{
    std::vector<uint32_t> token_ids = tokenize(input);
    ggml_context_ptr ctx = init_engine(loader_->get_tensor_count());
    std::vector<std::unique_ptr<Layer>> layers = build_graph_(ctx.get());
    std::vector<Layer *> layer_ptrs;
    layer_ptrs.reserve(layers.size());
    for (auto &l : layers)
    {
        layer_ptrs.push_back(l.get());
    }
    Graph graph(ctx.get(), layer_ptrs);
    ggml_cgraph *gf = graph.build(token_ids.size());
    ggml_graph_print(gf);
}
