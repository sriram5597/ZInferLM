#include "zinferlm/models.h"
#include "zinferlm/tokenizer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0] << " <model_path> -f <filename>\n";
    return 1;
  }

  std::string model_path = argv[1];
  std::string filename;

  for (int i = 2; i < argc; i++)
  {
    std::string arg = argv[i];
    if (arg == "-f" && i + 1 < argc)
    {
      filename = argv[++i];
    }
  }

  if (filename.empty())
  {
    std::cerr << "Error: -f <filename> is required\n";
    return 1;
  }

  std::ifstream file(filename);
  if (!file.is_open())
  {
    std::cerr << "Error: Could not open file: " << filename << "\n";
    return 1;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string prompt = buffer.str();

  auto is_loaded = zinferlm::Model::load(model_path.c_str());
  if (!is_loaded)
  {
    std::cerr << "Failed to load model: " << model_path << "\n";
    return 1;
  }

  auto &model = zinferlm::Model::instance();
  zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(model);

  auto token_ids = tokenizer.tokenize(prompt);

  for (auto token_id : token_ids)
  {
    std::string token = tokenizer.decode(token_id);
    std::cout << token_id << " -> \'" << token << "\'" << "\n";
  }

  return 0;
}
