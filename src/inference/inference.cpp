#include <cstdint>
#include <iostream>

#include <ggml.h>
#include <stdarg.h>
#include <string>
#include <vector>
#include <zinferlm/inference.h>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

#include "sampling/samplers.h"

std::string zinferlm::Inference::invoke(std::string input) {
  zinferlm::Model &model = zinferlm::Model::instance();
  zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(model);
  std::string output = "";
  std::string template_str = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>\nSay Hi<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int32_t> tokens = {151643, 13048, };
  for (int i = 0; i < 10; i++) {
    // std::vector<int32_t> tokens = tokenizer.tokenize(input);
    // tokens.emplace(tokens.begin(), 151643);
    std::vector<float> logits = model.invoke(tokens);
    sampler_params_t params = {.temperature = 0.2, .top_k = 0};
    Sampler sampler(params);
    std::pair<uint64_t, float> sample = sampler.sample(logits);
    tokens.push_back(sample.first);
    output += tokenizer.decode(static_cast<uint32_t>(sample.first));
  }
  return output;
}

std::vector<int32_t> zinferlm::Inference::tokenize(std::string input) {
  zinferlm::Model &model = zinferlm::Model::instance();
  zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(model);
  return tokenizer.tokenize(input);
}
