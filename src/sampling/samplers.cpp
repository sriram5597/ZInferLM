#include "samplers.h"
#include "logging/log.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <ostream>
#include <random>
#include <utility>

namespace zinferlm {
namespace internal {

Sampler::Sampler() : temp_(0.0), top_k_(0) {}
Sampler::Sampler(sampler_params_t params)
    : temp_(params.temperature), top_k_(params.top_k) {}

std::pair<uint64_t, float> Sampler::greedy_(std::span<const float> logits) {
  auto max_logit = std::max_element(logits.begin(), logits.end());
  uint64_t max_ele = std::distance(logits.begin(), max_logit);
  return std::pair<uint64_t, float>(max_ele, *max_logit);
}

std::vector<float> Sampler::softmax_(std::span<const float> logits) {
  float m = *std::max_element(logits.begin(), logits.end());
  std::vector<float> p(logits.size());
  double sum = 0.0;
  for (size_t i = 0; i < logits.size(); i++) {
    p[i] = std::exp(logits[i] - m);
    sum += p[i];
  }
  for (auto &v : p)
    v /= sum;
  return p;
}

std::pair<uint64_t, float> Sampler::top_k_sample_(std::span<const float> logits) {
  size_t k = std::min<size_t>(top_k_, logits.size());
  std::vector<float> vals(logits.begin(), logits.end());
  std::partial_sort(vals.begin(), vals.begin() + k, vals.end(),
                    std::greater<float>());
  float threshold = vals[k - 1];
  for (auto &l : vals) {
    if (l < threshold)
      l = -INFINITY;
  }
  auto probs = softmax_(vals);
  double cum = 0.0;

  static thread_local std::mt19937 rng{std::random_device{}()};
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  double rand = dist(rng);
  for (size_t i = 0; i < probs.size(); i++) {
    if (probs[i] <= 0.0f)
      continue;
    cum += probs[i];
    if (cum >= rand) {
      return std::pair<uint64_t, float>{i, probs[i]};
    }
  }
  return std::pair<uint64_t, float>{logits.size() - 1, probs.back()};
}

std::pair<uint64_t, float> Sampler::top_p_sample_(std::span<const float> logits) {
  auto probs = softmax_(logits);
  std::vector<std::pair<uint64_t, float>> prob_pairs;
  for (size_t i = 0; i < probs.size(); i++) {
    prob_pairs.push_back({i, probs[i]});
  }
  std::sort(prob_pairs.begin(), prob_pairs.end(),
            [](auto a, auto b) { return a.second > b.second; });
  float cum = 0.0;
  size_t cutoff_ind = logits.size();
  for (size_t i = 0; i < prob_pairs.size(); i++) {
    cum += prob_pairs[i].second;
    if (cum >= top_p_) {
      cutoff_ind = i + 1;
      break;
    }
  }
  prob_pairs.resize(cutoff_ind);
  LOG_DBG("Cutoff size: %zu\n", cutoff_ind);
  std::vector<double> weights(prob_pairs.size());
  for (size_t i = 0; i < weights.size(); i++) {
    weights[i] = static_cast<double>(prob_pairs[i].second / cum);
  }
  static thread_local std::mt19937 rng{std::random_device{}()};
  std::discrete_distribution<uint32_t> dist(weights.begin(), weights.end());
  return prob_pairs[dist(rng)];
}

std::pair<uint64_t, float> Sampler::sample(std::span<const float> logits) {
  if (logits.empty()) {
    return std::pair<uint64_t, float>{0, 0.0f};
  }

  if (top_k_ > 0) {
    if (temp_ > 0.0f) {
      std::vector<float> scaled(logits.begin(), logits.end());
      for (auto &l : scaled) {
        l /= temp_;
      }
      return top_k_sample_(scaled);
    }
    return top_k_sample_(logits);
  }

  if (temp_ > 0.0f) {
    std::vector<float> scaled(logits.begin(), logits.end());
    for (auto &l : scaled) {
      l /= temp_;
    }
    return greedy_(scaled);
  }
  return greedy_(logits);
}

} // namespace internal
} // namespace zinferlm
