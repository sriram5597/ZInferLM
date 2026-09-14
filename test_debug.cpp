#include <iostream>
#include <vector>
#include <zinferlm/models.h>

int main() {
    std::cout << "Loading model..." << std::endl;
    auto is_loaded = zinferlm::Model::load("model_files/qwen2.5-1.5b-instruct-q5_k_m.gguf");
    if (!is_loaded) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    std::cout << "Model loaded, running inference..." << std::endl;
    auto &model = zinferlm::Model::instance();
    
    // Test with token 13048 ("Hi")
    std::vector<uint32_t> tokens = {13048};
    std::cout << "Input tokens: ";
    for (auto t : tokens) std::cout << t << " ";
    std::cout << std::endl;
    
    auto logits = model.invoke(tokens);
    
    std::cout << "Inference complete, got " << logits.size() << " logits" << std::endl;
    std::cout << "First 10 logits: ";
    for (int i = 0; i < 10 && i < logits.size(); i++) {
        std::cout << logits[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
