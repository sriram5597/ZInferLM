#pragma once
#include <cstdint>
#include <memory>
#include <ggml.h>

#include "metadata.h"

class TensorInfo
{
private:
    std::unique_ptr<MetadataStringValue> name_;
    const uint32_t *n_dimensions_;
    std::vector<const uint64_t *> dimensions_;
    const ggml_type *ggml_type_;
    const uint64_t *offset_;
    int size_;

    TensorInfo(std::unique_ptr<MetadataStringValue> n, const uint32_t *nd, const ggml_type *t, const uint64_t *o) : name_(std::move(n)), n_dimensions_(nd), ggml_type_(t), offset_(o) {}

public:
    static TensorInfo from_ptr(const char *ptr);
    int size() const;
    ggml_type type() const;
    std::string name_string() const;
    std::string n_dim_sting() const;
    std::string dims_string() const;
    std::string type_string() const;
    std::string offset_string() const;
};
