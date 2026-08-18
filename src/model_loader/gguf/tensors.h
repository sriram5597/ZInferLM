#pragma once
#include <cstdint>
#include <memory>
#include <ggml.h>

#include "metadata.h"

class TensorInfo
{
private:
    TensorInfo(std::unique_ptr<MetadataStringValue> n, const uint32_t *nd, const ggml_type *t, const uint64_t *o) : name(std::move(n)), n_dimensions(nd), dtype(t), offset(o) {}

public:
    std::unique_ptr<MetadataStringValue> name;
    const uint32_t *n_dimensions;
    std::vector<uint64_t> dimensions;
    const ggml_type *dtype;
    const uint64_t *offset;

    static TensorInfo from_ptr(const char *ptr);
    int size() const;
};
