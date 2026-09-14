#include <cstdint>
#include <iostream>
#include <vector>
#include <zinferlm/models.h>
#include <ggml.h>

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
  
  // Get token embedding weight info
  auto tensor_infos = model.tensor_info();
  for (const auto &info : tensor_infos) {
    if (info.name.find("token_embd") != std::string::npos) {
      std::cout << "Found: " << info.name << " | type: " << info.type_name 
                << " | dims: [";
      for (size_t i = 0; i < info.dimensions.size(); i++) {
        std::cout << info.dimensions[i];
        if (i < info.dimensions.size() - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      
      // Read first few values from the embedding for token 13048
      if (info.dimensions.size() >= 2) {
        size_t emb_dim = info.dimensions[0];
        size_t vocab_size = info.dimensions[1];
        std::cout << "emb_dim: " << emb_dim << ", vocab_size: " << vocab_size << std::endl;
        
        // Token 13048
        size_t token_id = 13048;
        size_t offset = token_id * emb_dim * ggml_type_size(static_cast<ggml_type>(info.type_id));
        
        std::vector<float> emb_values(10);
        // Note: This won't work directly for quantized types, but let's see
        std::cout << "Token " << token_id << " embedding (first 10 values, raw): ";
        // We can't easily read quantized embeddings without dequantization
      }
    }
  }
  
  return 0;
}
