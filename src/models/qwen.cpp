#include <zinferlm/models.h>
#include <iostream>

#include "qwen.h"

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