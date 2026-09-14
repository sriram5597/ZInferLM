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
  for (int i = 0; i < 1; i++) {
    // std::vector<uint32_t> tokens = tokenizer.tokenize(template_str + output);
    // for (int i = 0; i < tokens.size(); i++) {
    //   std::cout << tokens[i] << " ";
    // }
    // std::cout << std::endl;
    std::vector<uint32_t> tokens = {151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 198, 45764, 21018, 151645, 198, 151644, 77091, 198};
    std::vector<float> logits = model.invoke(tokens);
    sampler_params_t params = {.temperature = 1.0, .top_k = 0};
    Sampler sampler(params);
    std::pair<uint64_t, float> sample = sampler.sample(logits);
    output += tokenizer.decode(static_cast<uint32_t>(sample.first));
  }
  return output;
}

std::vector<uint32_t> zinferlm::Inference::tokenize(std::string input) {
  zinferlm::Model &model = zinferlm::Model::instance();
  zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(model);
  return tokenizer.tokenize(input);
}
