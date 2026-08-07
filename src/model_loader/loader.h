#pragma once

#include <zinferlm/models.h>

class ModelLoader
{
public:
    virtual zinferlm::model_info_t info() const = 0;
    virtual zinferlm::tokenizer_info_t tokenizer_info() const = 0;
    virtual std::vector<zinferlm::tensor_info_t> tensor_info() const = 0;
};