#include <ggml-cpp.h>
#include <memory>
#include <functional>
#include <cstdint>
#include <string>
#include <vector>

#include "layers/layers.h"

class Graph
{
private:
    ggml_context *ctx_;
    std::vector<Layer *> layers_;
    ggml_tensor *input_;
    ggml_tensor *output_;
    ggml_backend_ptr backend_;
    ggml_backend_sched_ptr sched_;
    std::vector<ggml_tensor*> intermediate_tensors_;
    
    // Debug logging support
    bool debug_mode_ = false;

public:
    Graph(ggml_context *c, std::vector<Layer *> l);
    ggml_cgraph *build(uint32_t n_tokens);
    ggml_tensor *execute(std::vector<int32_t> input);
    
    void set_debug_mode(bool enabled) { debug_mode_ = enabled; }
};

ggml_context_ptr init_engine(uint64_t n_op_estimate);
