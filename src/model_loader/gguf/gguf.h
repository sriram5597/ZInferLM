#pragma once

#include <cstdint>
#include <functional>
#include <variant>
#include <vector>
#include <unordered_map>
#include <string_view>
#include <zinferlm/models.h>
#include "model_loader/loader.h"
#include "metadata.h"
#include "tensors.h"

// Header metadata key: "general.file_type"
enum class gguf_file_type : uint32_t
{
  ALL_F32 = 0,         // All unquantized 32-bit floats [cite: 43]
  MOSTLY_F16 = 1,      // Mostly 16-bit float weights [cite: 44]
  MOSTLY_Q4_0 = 2,     // Mostly 4-bit uniform quantized (Q4_0) [cite: 45]
  MOSTLY_Q4_1 = 3,     // Mostly 4-bit uniform quantized with zero-point (Q4_1) [cite: 45, 46]
  MOSTLY_Q8_0 = 7,     // Mostly 8-bit uniform quantized (Q8_0) [cite: 46]
  MOSTLY_Q5_0 = 8,     // Mostly 5-bit uniform quantized (Q5_0) [cite: 47]
  MOSTLY_Q5_1 = 9,     // Mostly 5-bit uniform quantized with zero-point (Q5_1) [cite: 47, 48]
  MOSTLY_Q2_K = 10,    // Mostly 2-bit K-quantized (Q2_K) [cite: 48]
  MOSTLY_Q3_K_S = 11,  // Mostly 3-bit Small K-quantized (Q3_K_S) [cite: 49]
  MOSTLY_Q3_K_M = 12,  // Mostly 3-bit Medium K-quantized (Q3_K_M) [cite: 49]
  MOSTLY_Q3_K_L = 13,  // Mostly 3-bit Large K-quantized (Q3_K_L) [cite: 50]
  MOSTLY_Q4_K_S = 14,  // Mostly 4-bit Small K-quantized (Q4_K_S) [cite: 50]
  MOSTLY_Q4_K_M = 15,  // Mostly 4-bit Medium K-quantized (Q4_K_M) [cite: 51]
  MOSTLY_Q5_K_S = 16,  // Mostly 5-bit Small K-quantized (Q5_K_S) [cite: 51]
  MOSTLY_Q5_K_M = 17,  // Mostly 5-bit Medium K-quantized (Q5_K_M) [cite: 52]
  MOSTLY_Q6_K = 18,    // Mostly 6-bit K-quantized (Q6_K) [cite: 52, 53]
  MOSTLY_IQ2_XXS = 19, // Mostly 2-bit Importance Matrix (Extra Extra Small) [cite: 53]
  MOSTLY_IQ2_XS = 20,  // Mostly 2-bit Importance Matrix (Extra Small) [cite: 54]
  MOSTLY_Q2_K_S = 21,  // Mostly 2-bit Small K-quantized [cite: 54, 55]
  MOSTLY_IQ3_XS = 22,  // Mostly 3-bit Importance Matrix (Extra Small) [cite: 55]
  MOSTLY_IQ3_XXS = 23  // Mostly 3-bit Importance Matrix (Extra Extra Small) [cite: 55, 56]
};

// Helper function to convert GGUF file type to string
constexpr const char *gguf_file_type_name(gguf_file_type type)
{
  switch (type)
  {
  case gguf_file_type::ALL_F32:
    return "ALL_F32";
  case gguf_file_type::MOSTLY_F16:
    return "MOSTLY_F16";
  case gguf_file_type::MOSTLY_Q4_0:
    return "MOSTLY_Q4_0";
  case gguf_file_type::MOSTLY_Q4_1:
    return "MOSTLY_Q4_1";
  case gguf_file_type::MOSTLY_Q8_0:
    return "MOSTLY_Q8_0";
  case gguf_file_type::MOSTLY_Q5_0:
    return "MOSTLY_Q5_0";
  case gguf_file_type::MOSTLY_Q5_1:
    return "MOSTLY_Q5_1";
  case gguf_file_type::MOSTLY_Q2_K:
    return "MOSTLY_Q2_K";
  case gguf_file_type::MOSTLY_Q3_K_S:
    return "MOSTLY_Q3_K_S";
  case gguf_file_type::MOSTLY_Q3_K_M:
    return "MOSTLY_Q3_K_M";
  case gguf_file_type::MOSTLY_Q3_K_L:
    return "MOSTLY_Q3_K_L";
  case gguf_file_type::MOSTLY_Q4_K_S:
    return "MOSTLY_Q4_K_S";
  case gguf_file_type::MOSTLY_Q4_K_M:
    return "MOSTLY_Q4_K_M";
  case gguf_file_type::MOSTLY_Q5_K_S:
    return "MOSTLY_Q5_K_S";
  case gguf_file_type::MOSTLY_Q5_K_M:
    return "MOSTLY_Q5_K_M";
  case gguf_file_type::MOSTLY_Q6_K:
    return "MOSTLY_Q6_K";
  case gguf_file_type::MOSTLY_IQ2_XXS:
    return "MOSTLY_IQ2_XXS";
  case gguf_file_type::MOSTLY_IQ2_XS:
    return "MOSTLY_IQ2_XS";
  case gguf_file_type::MOSTLY_Q2_K_S:
    return "MOSTLY_Q2_K_S";
  case gguf_file_type::MOSTLY_IQ3_XS:
    return "MOSTLY_IQ3_XS";
  case gguf_file_type::MOSTLY_IQ3_XXS:
    return "MOSTLY_IQ3_XXS";
  default:
    return "UNKNOWN";
  }
}

struct gguf_header_t
{
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
};

using metadata_map_t = std::unordered_map<std::string, std::unique_ptr<Metadata>>;

class GGUFLoader : public ModelLoader
{
public:
  const gguf_header_t *header;
  metadata_map_t metadata;
  std::vector<std::unique_ptr<TensorInfo>> tensors;
  Metadata *get_metadata(std::string key) const;
  zinferlm::model_info_t info() const override;
  zinferlm::tokenizer_info_t tokenizer_info() const override;
  std::vector<zinferlm::tensor_info_t> tensor_info() const override;

  void set_mapped_memory(std::unique_ptr<const char, std::function<void(const char *)>> mem)
  {
    mapped_memory_ = std::move(mem);
  }

private:
  std::unique_ptr<const char, std::function<void(const char *)>> mapped_memory_;
  mutable Metadata default_metadata_ = Metadata::empty();
};

bool is_gguf_file(int *);
std::unique_ptr<GGUFLoader> load_gguf_file(int *, size_t);
