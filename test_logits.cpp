#include <cstdint>
#include <iostream>
#include <vector>
#include <zinferlm/models.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path>\n";
    return 1;
  }
  
  auto is_loaded = zinferlm::Model::load(argv[1]);
  if (!is_loaded) {
    std::cerr << "Failed to load model\n";
    return 1;
  }
  
  auto &model = zinferlm::Model::instance();
  
  // Token 13048 is "Hi" from llama.cpp tokenizer
  std::vector<uint32_t> tokens = {13048};
  std::cout << "Running with token: 13048 (Hi)\n";
  
  std::vector<float> logits = model.invoke(tokens);
  
  std::cout << "\nFirst 20 logits from inferlm:\n";
  for (int i = 0; i < 20 && i < logits.size(); i++) {
    std::cout << i << ": " << logits[i] << "\n";
  }
  
  // Find max logit
  float max_val = logits[0];
  int max_idx = 0;
  for (int i = 1; i < logits.size(); i++) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      max_idx = i;
    }
  }
  std::cout << "\nMax logit at index " << max_idx << ": " << max_val << "\n";
  
  return 0;
}
