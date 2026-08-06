#include <zinferlm/model_loader.h>
#include <iostream>
#include "gguf.h"

zinferlm::model_info_t GGUFModel::info() const
{
    zinferlm::model_info_t model_info;
    auto it = file_->metadata.find("general.architecture");
    if (it != file_->metadata.end())
    {
        model_info.architecture = it->second->value_string();
    }
    auto v = file_->metadata.find("general.version");
    if (v != file_->metadata.end())
    {
        model_info.version = v->second->value_string();
    }
    it = file_->metadata.find("general.name");
    if (it != file_->metadata.end())
    {
        model_info.name = it->second->value_string();
    }
    it = file_->metadata.find("general.file_type");
    if (it != file_->metadata.end())
    {
        if (const MetadataValue *val = it->second->value())
        {
            if (const MetadataPrimitiveValue *prim = dynamic_cast<const MetadataPrimitiveValue *>(val))
            {
                gguf_file_type t = static_cast<gguf_file_type>(std::get<uint32_t>(prim->value()));
                model_info.file_type = gguf_file_type_name(t);
            }
        }
    }

    return model_info;
}

zinferlm::tokenizer_info_t GGUFModel::tokenizer_info() const
{
    zinferlm::tokenizer_info_t t_info;

    auto it = file_->metadata.find("tokenizer.ggml.model");
    if (it != file_->metadata.end())
    {
        t_info.model = it->second->value_string();
    }
    it = file_->metadata.find("tokenizer.ggml.pre");
    if (it != file_->metadata.end())
    {
        t_info.pre = it->second->value_string();
    }
    it = file_->metadata.find("tokenizer.ggml.tokens");
    if (it != file_->metadata.end())
    {
        if (const MetadataArrayValue *arr = dynamic_cast<const MetadataArrayValue *>(it->second->value()))
        {
            for (auto &&t : arr->elements())
            {
                t_info.tokens.push_back(t->to_string());
            }
        }
    }
    it = file_->metadata.find("tokenizer.ggml.merges");
    if (it != file_->metadata.end())
    {
        if (const MetadataArrayValue *arr = dynamic_cast<const MetadataArrayValue *>(it->second->value()))
        {
            for (auto &&t : arr->elements())
            {
                t_info.merges.push_back(t->to_string());
            }
        }
    }
    it = file_->metadata.find("tokenizer.ggml.token_type");
    if (it != file_->metadata.end())
    {
        if (const MetadataArrayValue *arr = dynamic_cast<const MetadataArrayValue *>(it->second->value()))
        {
            for (auto &&t : arr->elements())
            {
                const MetadataPrimitiveValue *val = dynamic_cast<const MetadataPrimitiveValue *>(t.get());
                t_info.token_type.push_back(std::get<int32_t>(val->value()));
            }
        }
    }
    return t_info;
}