#pragma once
#include <cstdint>
#include <memory>

#include "metadata.h"

enum GGMLType : uint32_t
{
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0 = 34,
    GGML_TYPE_TQ2_0 = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_MXFP4 = 39, // MXFP4 (1 block)
    GGML_TYPE_COUNT = 40,
};

constexpr std::string_view ggml_type_name(GGMLType t)
{
    switch (t)
    {
    case GGML_TYPE_F32:     return "F32";
    case GGML_TYPE_F16:     return "F16";
    case GGML_TYPE_Q4_0:    return "Q4_0";
    case GGML_TYPE_Q4_1:    return "Q4_1";
    case GGML_TYPE_Q5_0:    return "Q5_0";
    case GGML_TYPE_Q5_1:    return "Q5_1";
    case GGML_TYPE_Q8_0:    return "Q8_0";
    case GGML_TYPE_Q8_1:    return "Q8_1";
    case GGML_TYPE_Q2_K:    return "Q2_K";
    case GGML_TYPE_Q3_K:    return "Q3_K";
    case GGML_TYPE_Q4_K:    return "Q4_K";
    case GGML_TYPE_Q5_K:    return "Q5_K";
    case GGML_TYPE_Q6_K:    return "Q6_K";
    case GGML_TYPE_Q8_K:    return "Q8_K";
    case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
    case GGML_TYPE_IQ2_XS:  return "IQ2_XS";
    case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
    case GGML_TYPE_IQ1_S:   return "IQ1_S";
    case GGML_TYPE_IQ4_NL:  return "IQ4_NL";
    case GGML_TYPE_IQ3_S:   return "IQ3_S";
    case GGML_TYPE_IQ2_S:   return "IQ2_S";
    case GGML_TYPE_IQ4_XS:  return "IQ4_XS";
    case GGML_TYPE_I8:      return "I8";
    case GGML_TYPE_I16:     return "I16";
    case GGML_TYPE_I32:     return "I32";
    case GGML_TYPE_I64:     return "I64";
    case GGML_TYPE_F64:     return "F64";
    case GGML_TYPE_IQ1_M:   return "IQ1_M";
    case GGML_TYPE_BF16:    return "BF16";
    case GGML_TYPE_TQ1_0:   return "TQ1_0";
    case GGML_TYPE_TQ2_0:   return "TQ2_0";
    case GGML_TYPE_MXFP4:   return "MXFP4";
    default:                return "UNKNOWN";
    }
}

class TensorInfo
{
private:
    std::unique_ptr<MetadataStringValue> name_;
    const uint32_t *n_dimensions_;
    std::vector<const uint64_t *> dimensions_;
    const GGMLType *ggml_type_;
    const uint64_t *offset_;
    int size_;

    TensorInfo(std::unique_ptr<MetadataStringValue> n, const uint32_t *nd, const GGMLType *t, const uint64_t *o) : name_(std::move(n)), n_dimensions_(nd), ggml_type_(t), offset_(o) {}

public:
    static TensorInfo from_ptr(const char *ptr);
    int size() const;
    GGMLType type() const;
    std::string name_string() const;
    std::string n_dim_sting() const;
    std::string dims_string() const;
    std::string offset_string() const;
};
