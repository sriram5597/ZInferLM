#include <cstdint>
#include <iostream>
#include <vector>
#include <zinferlm/models.h>
#include <ggml.h>
#include <ggml-backend.h>
#include <fstream>

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
  
  // Just run with token 13048 and print all logits
  std::vector<uint32_t> tokens = {13048};
  std::vector<float> logits = model.invoke(tokens);
  
  // Save logits to file for comparison
  std::cout << "Saving " << logits.size() << " logits to inferlm_logits.txt\n";
  std::ofstream f("inferlm_logits.txt");
  for (int i = 0; i < logits.size(); i++) {
    f << i << ": " << logits[i] << "\n";
  }
  f.close();
  
  // Print first 20 and last 20
  std::cout << "First 20 logits: ";
  for (int i = 0; i < 20; i++) std::cout << logits[i] << " ";
  std::cout << "\n";
  
  // Find max
  float max_val = logits[0];
  int max_idx = 0;
  for (int i = 1; i < logits.size(); i++) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      max_idx = i;
    }
  }
  std::cout << "Max logit at index " << max_idx << ": " << max_val << "\n";
  
  return 0;
}
