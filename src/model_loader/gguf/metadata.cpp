#include <iostream>
#include <string>
#include <format>
#include <memory>
#include <vector>
#include <type_traits>
#include "metadata.h"
std::unique_ptr<MetadataValue> MetadataValue::from_ptr(const char *ptr, MetadataDType dtype)
{
    switch (dtype)
    {
    case MetadataDType::GGUF_METADATA_VALUE_TYPE_STRING:
        return std::make_unique<MetadataStringValue>(MetadataStringValue::from_ptr(ptr));
        break;
    case MetadataDType::GGUF_METADATA_VALUE_TYPE_ARRAY:
        return std::make_unique<MetadataArrayValue>(MetadataArrayValue::from_ptr(ptr));
    default:
        return std::make_unique<MetadataPrimitiveValue>(MetadataPrimitiveValue::from_ptr(ptr, dtype));
    }
}

std::string MetadataPrimitiveValue::to_string() const
{
    return std::visit([]<typename T>(const T &arg) -> std::string
                      {
        if constexpr (std::is_same_v<T, bool>) {
            return arg ? "true" : "false";
        } else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
            return std::to_string(static_cast<int>(arg));
        } else {
            return std::format("{}", arg);
    } }, data_);
}

primitive_value_t MetadataPrimitiveValue::value() const
{
    return data_;
}

MetadataPrimitiveValue MetadataPrimitiveValue::from_ptr(const char *ptr, MetadataDType dtype)
{
    primitive_value_t val;
    int size = 0;
    switch (dtype)
    {
    case GGUF_METADATA_VALUE_TYPE_UINT8:
        val = *reinterpret_cast<const uint8_t *>(ptr);
        size = sizeof(uint8_t);
        break;
    case GGUF_METADATA_VALUE_TYPE_INT8:
        val = *reinterpret_cast<const int8_t *>(ptr);
        size = sizeof(int8_t);
        break;
    case GGUF_METADATA_VALUE_TYPE_UINT16:
        val = *reinterpret_cast<const uint16_t *>(ptr);
        size = sizeof(uint16_t);
        break;
    case GGUF_METADATA_VALUE_TYPE_INT16:
        val = *reinterpret_cast<const int16_t *>(ptr);
        size = sizeof(int16_t);
        break;
    case GGUF_METADATA_VALUE_TYPE_UINT32:
        val = *reinterpret_cast<const uint32_t *>(ptr);
        size = sizeof(uint32_t);
        break;
    case GGUF_METADATA_VALUE_TYPE_INT32:
        val = *reinterpret_cast<const int32_t *>(ptr);
        size = sizeof(int32_t);
        break;
    case GGUF_METADATA_VALUE_TYPE_FLOAT32:
        val = *reinterpret_cast<const float *>(ptr);
        size = sizeof(float);
        break;
    case GGUF_METADATA_VALUE_TYPE_BOOL:
        val = *reinterpret_cast<const bool *>(ptr);
        size = sizeof(bool);
        break;
    case GGUF_METADATA_VALUE_TYPE_UINT64:
        val = *reinterpret_cast<const uint64_t *>(ptr);
        size = sizeof(uint64_t);
        break;
    case GGUF_METADATA_VALUE_TYPE_INT64:
        val = *reinterpret_cast<const int64_t *>(ptr);
        size = sizeof(int64_t);
        break;
    case GGUF_METADATA_VALUE_TYPE_FLOAT64:
        val = *reinterpret_cast<const double *>(ptr);
        size = sizeof(double);
        break;
    }
    return MetadataPrimitiveValue(val, size);
}

int MetadataPrimitiveValue::size() const
{
    return size_;
}

std::string MetadataStringValue::to_string() const
{
    return std::string(data_);
}

int MetadataStringValue::size() const
{
    return sizeof(*len_) + *len_;
}

MetadataStringValue MetadataStringValue::from_ptr(const char *ptr)
{
    const uint64_t *len = reinterpret_cast<const uint64_t *>(ptr);
    std::string_view val(ptr + sizeof(*len), *len);

    return MetadataStringValue(val, len);
}

MetadataArrayValue MetadataArrayValue::from_ptr(const char *ptr)
{
    const char *cur = ptr;
    const MetadataDType *dtype = reinterpret_cast<const MetadataDType *>(cur);
    cur += sizeof(*dtype);
    const uint64_t *len = reinterpret_cast<const uint64_t *>(cur);
    cur += sizeof(*len);
    MetadataArrayValue val = MetadataArrayValue(dtype, len);
    for (uint64_t i = 0; i < *len; i++)
    {
        std::unique_ptr<MetadataValue> ele = MetadataValue::from_ptr(cur, *dtype);
        cur += ele->size();
        val.elements_.push_back(std::move(ele));
    }
    return val;
}

int MetadataArrayValue::size() const
{
    int val_size = 0;

    for (const auto &e : elements_)
    {
        val_size += e->size();
    }
    return sizeof(*val_type_) + sizeof(*len_) + val_size;
}

const std::vector<std::unique_ptr<MetadataValue>> &MetadataArrayValue::elements() const
{
    return elements_;
}

std::string MetadataArrayValue::to_string() const
{
    std::string res = "[ ";
    for (int i = 0; i < 50; i++)
    {
        if (i > 0)
        {
            res += ", ";
        }
        res += elements_[i]->to_string();
    }
    if (elements_.size() > 50)
    {
        res += " ... ";
    }
    res += " ]";
    return res;
}

Metadata Metadata::from_ptr(const char *ptr)
{
    const char *cur = ptr;
    const uint64_t *len = reinterpret_cast<const uint64_t *>(ptr);
    MetadataStringValue key = MetadataStringValue::from_ptr(cur);
    cur += key.size();
    const MetadataDType val_type = *(reinterpret_cast<const MetadataDType *>(cur));
    cur += sizeof(val_type);
    std::unique_ptr<MetadataValue> val = MetadataValue::from_ptr(cur, val_type);
    return Metadata(key, val_type, std::move(val));
}

std::string Metadata::to_string() const
{
    return std::format("Key: {}, Value: {}", key_.to_string(), val_->to_string());
}

int Metadata::size() const
{
    return key_.size() + sizeof(val_type_) + val_->size();
}