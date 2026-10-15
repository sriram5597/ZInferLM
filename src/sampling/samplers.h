#pragma once

#include <cstdint>
#include <vector>

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
  std::pair<uint64_t, float> greedy_(std::vector<float> logits);
  std::pair<uint64_t, float> top_k_sample_(std::vector<float> logits);
  std::vector<float> softmax_(std::vector<float> logits);
  std::pair<uint64_t, float> top_p_sample_(std::vector<float> logits);

public:
  Sampler(sampler_params_t params);
  Sampler();
  std::pair<uint64_t, float> sample(std::vector<float> logits);
};
