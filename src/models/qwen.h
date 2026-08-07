#include <memory>
#include <zinferlm/models.h>

#include "model_loader/loader.h"

class QwenModel : public zinferlm::Model
{
private:
    std::unique_ptr<ModelLoader> loader_;

public:
    QwenModel(std::unique_ptr<ModelLoader> f) : loader_(std::move(f)) {}
    zinferlm::model_info_t info() const override;
    zinferlm::tokenizer_info_t tokenizer_info() const override;
    std::vector<zinferlm::tensor_info_t> tensor_info() const override;
};