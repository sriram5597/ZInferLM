#include <ggml-cpp.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <cstdint>
#include <cstring>

#include "layers/layers.h"
#include "graph.h"

ggml_context_ptr init_engine(uint64_t tensor_count)
{
    uint64_t ctx_size = tensor_count * ggml_tensor_overhead() + ggml_graph_overhead() + 2048;
    ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc = true};
    return ggml_context_ptr{ggml_init(params)};
}

ggml_cgraph *Graph::build(uint32_t input_size)
{
    ggml_cgraph *gf = ggml_new_graph(ctx_);
    input_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, input_size);
    ggml_set_name(input_, "input");
    ggml_tensor *current = input_;

    for (auto l : layers_)
    {
        current = (*l)(current);
    }
    output_ = current;
    ggml_set_name(output_, "output");
    ggml_build_forward_expand(gf, output_);
    return gf;
}

ggml_tensor *Graph::execute(std::vector<uint32_t> input)
{
    ggml_cgraph *gf = build(input.size());
    ggml_backend_ptr backend{ggml_backend_cpu_init()};
    ggml_gallocr_ptr gallocr{ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get()))};
    ggml_gallocr_reserve(gallocr.get(), gf);
    ggml_gallocr_alloc_graph(gallocr.get(), gf);

    std::memcpy(input_->data, input.data(), input.size() * sizeof(uint32_t));

    ggml_backend_graph_compute(backend.get(), gf);
    return output_;
}
