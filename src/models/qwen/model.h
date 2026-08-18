#include <memory>
#include <ggml.h>
#include <ggml-cpp.h>
#include <memory>
#include <zinferlm/models.h>

#include "layers/layers.h"
#include "model_loader/loader.h"

class QwenModel : public zinferlm::Model
{
private:
    std::unique_ptr<ModelLoader> loader_;
    ggml_context_ptr ctx_;
    std::vector<std::unique_ptr<Layer>> build_graph_(ggml_context *ctx);

public:
    QwenModel(std::unique_ptr<ModelLoader> f) : loader_(std::move(f)) {}
    zinferlm::model_info_t info() const override;
    zinferlm::tokenizer_info_t tokenizer_info() const override;
    std::vector<zinferlm::tensor_info_t> tensor_info() const override;
    std::string invoke(std::string input) override;
    std::vector<uint32_t> tokenize(std::string input) override;
    void summary(std::string input) override;
};
