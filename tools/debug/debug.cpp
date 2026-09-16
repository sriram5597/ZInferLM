#include "zinferlm/inference.h"
#include <iostream>
#include <string>
#include <zinferlm/models.h>

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <model_path> [prompt]\n";
    return 1;
  }

  std::string model_path = argv[1];
  std::string prompt = (argc >= 3) ? argv[2] : "";

  auto is_loaded = zinferlm::Model::load(model_path.c_str());
  if (!is_loaded)
  {
    std::cerr << "Failed to load model: " << model_path << "\n";
    return 1;
  }

  auto &model = zinferlm::Model::instance();
  model.set_debug(true);

  zinferlm::Inference infer;
  std::string output = infer.invoke(prompt);
  std::cout << output << std::endl;

  return 0;
}
