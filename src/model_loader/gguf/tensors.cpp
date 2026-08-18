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

    std::vector<uint64_t> dimensions;
    for (int i = 0; i < *n_dim; i++)
    {
        const uint64_t *e = reinterpret_cast<const uint64_t *>(cur);
        dimensions.push_back(*e);
        cur += sizeof(*e);
    }

    const ggml_type *type = reinterpret_cast<const ggml_type *>(cur);
    cur += sizeof(*type);
    const uint64_t *offset = reinterpret_cast<const uint64_t *>(cur);
    cur += sizeof(*offset);
    TensorInfo info = TensorInfo(std::make_unique<MetadataStringValue>(val), n_dim, type, offset);
    info.dimensions = std::move(dimensions);
    return info;
}

int TensorInfo::size() const
{
    return name->size() + sizeof(*n_dimensions) + sizeof(uint64_t) * dimensions.size() + sizeof(*dtype) + sizeof(uint64_t);
}
