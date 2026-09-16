#include "samplers.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <utility>

Sampler::Sampler() : temp_(0.0) {}
Sampler::Sampler(sampler_params_t params)
    : temp_(params.temperature), top_k_(params.top_k) {}

std::pair<uint64_t, float> Sampler::greedy_(std::vector<float> logits) {
  auto max_logit = std::max_element(logits.begin(), logits.end());
  uint64_t max_ele = std::distance(logits.begin(), max_logit);
  return std::pair<uint64_t, float>(max_ele, *max_logit);
}

std::vector<float> Sampler::softmax_(std::vector<float> logits) {
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

std::pair<uint64_t, float> Sampler::top_k_sample_(std::vector<float> logits) {
  size_t k = std::min<size_t>(top_k_, logits.size());
  std::vector<float> vals(logits.begin(), logits.end());
  std::partial_sort(vals.begin(), vals.begin() + k, vals.end(),
                    std::greater<float>());
  float threshold = vals[k - 1];
  for (auto &l : logits) {
    if (l < threshold)
      l = -INFINITY;
  }
  auto probs = softmax_(logits);
  static thread_local std::mt19937 rng{std::random_device{}()};
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  double rand = dist(rng);
  double cum = 0.0;
  for (size_t i = 0; i < probs.size(); i++) {
    cum += probs[i];
    if (cum >= rand) {
      return std::pair<uint64_t, float>{i, probs[i]};
    }
  }
  return std::pair<uint64_t, float>{logits.size() - 1, probs.back()};
}

std::pair<uint64_t, float> Sampler::sample(std::vector<float> logits) {
  if (temp_ > 0.0) {
    for (auto &l : logits) {
      l /= temp_;
    }
  }

  if (top_k_ > 0)
    return top_k_sample_(logits);
  return greedy_(logits);
}
