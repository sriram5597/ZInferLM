#include <cstdint>
#include <memory>
#include <vector>

struct sampler_params_t {
  float temperature;
  int top_k;
};

class Sampler {
private:
  float temp_;
  int top_k_;
  std::pair<uint64_t, float> greedy_(std::vector<float> logits);
  std::pair<uint64_t, float> top_k_sample_(std::vector<float> logits);
  std::vector<float> softmax_(std::vector<float> logits);

public:
  Sampler(sampler_params_t params);
  Sampler();
  std::pair<uint64_t, float> sample(std::vector<float> logits);
};
