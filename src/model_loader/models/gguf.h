#include <memory>
#include <zinferlm/model_loader.h>

#include "model_loader/gguf/gguf.h"

class GGUFModel : public zinferlm::Model
{
private:
    std::unique_ptr<GGUFFile> file_;

public:
    GGUFModel(std::unique_ptr<GGUFFile> f) : file_(std::move(f)) {}
    zinferlm::model_info_t info() const override;
    zinferlm::tokenizer_info_t tokenizer_info() const override;
};