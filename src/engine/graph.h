#include <ggml-cpp.h>
#include <memory>
#include <functional>
#include <cstdint>

#include "layers/layers.h"

class Graph
{
private:
    ggml_context *ctx_;
    std::vector<Layer *> layers_;
    ggml_tensor *input_;
    ggml_tensor *output_;

public:
    Graph(ggml_context *c, std::vector<Layer *> l) : ctx_(c), layers_(l) {};
    ggml_tensor *output;
    ggml_cgraph *build(uint32_t n_tokens);
    ggml_tensor *execute(std::vector<uint32_t> input);
};

ggml_context_ptr init_engine(uint64_t tensor_count);