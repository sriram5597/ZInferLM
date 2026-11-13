#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace zinferlm {
namespace internal {

struct sampler_params_t {
  float temperature;
  uint32_t top_k;
  float top_p;
};

class Sampler {
private:
  float temp_;
  uint32_t top_k_;
  float top_p_;
  std::pair<uint64_t, float> greedy_(std::span<const float> logits);
  std::pair<uint64_t, float> top_k_sample_(std::span<const float> logits);
  std::vector<float> softmax_(std::span<const float> logits);
  std::pair<uint64_t, float> top_p_sample_(std::span<const float> logits);

public:
  Sampler(sampler_params_t params);
  Sampler();
  std::pair<uint64_t, float> sample(std::span<const float> logits);
};

} // namespace internal
} // namespace zinferlm
