#include <vector>
#include <memory>
#include <format>

#include "tensors.h"
#include "metadata.h"

TensorInfo TensorInfo::from_ptr(const char *ptr)
{
    const char *cur = ptr;
    MetadataStringValue val = MetadataStringValue::from_ptr(cur);
    cur += val.size();
    const uint32_t *n_dim = reinterpret_cast<const uint32_t *>(cur);
    cur += sizeof(*n_dim);

    std::vector<const uint64_t *> dimensions;
    for (int i = 0; i < *n_dim; i++)
    {
        const uint64_t *e = reinterpret_cast<const uint64_t *>(cur);
        dimensions.push_back(e);
        cur += sizeof(*e);
    }

    const ggml_type *type = reinterpret_cast<const ggml_type *>(cur);
    cur += sizeof(*type);
    const uint64_t *offset = reinterpret_cast<const uint64_t *>(cur);
    cur += sizeof(*offset);
    TensorInfo info = TensorInfo(std::make_unique<MetadataStringValue>(val), n_dim, type, offset);
    info.dimensions_ = std::move(dimensions);
    return info;
}

int TensorInfo::size() const
{
    return name_->size() + sizeof(*n_dimensions_) + sizeof(uint64_t) * dimensions_.size() + sizeof(*ggml_type_) + sizeof(uint64_t);
}

ggml_type TensorInfo::type() const
{
    return *ggml_type_;
}

std::string TensorInfo::name_string() const
{
    return name_->to_string();
}
std::string TensorInfo::n_dim_sting() const
{
    return std::format("{}", *n_dimensions_);
}

std::string TensorInfo::type_string() const
{
    return std::string(ggml_type_name(*ggml_type_));
}

std::string TensorInfo::dims_string() const
{
    std::string res = "[ ";
    for (auto d : dimensions_)
    {
        res += std::format("{}, ", *d);
    }
    res += " ]";
    return res;
}

std::string TensorInfo::offset_string() const
{
    return std::format("{}", *offset_);
}
