#include "zinferlm/models.h"
#include "zinferlm/tokenizer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "../log_flags.h"
#include "logging/log.h"

int main(int argc, char *argv[])
{
  auto args = zinferlm_tools::parse_log_flags(argc, argv);
  if (args.size() < 3)
  {
    LOG_ERR("Usage: %s <model_path> -f <filename> %s\n", argv[0],
            zinferlm_tools::log_flags_usage());
    return 1;
  }

  std::string model_path = args[0];
  std::string filename;

  for (size_t i = 1; i < args.size(); i++)
  {
    if (args[i] == "-f" && i + 1 < args.size())
    {
      filename = args[++i];
    }
  }

  if (filename.empty())
  {
    LOG_ERR("Error: -f <filename> is required\n");
    return 1;
  }

  std::ifstream file(filename);
  if (!file.is_open())
  {
    LOG_ERR("Error: Could not open file: %s\n", filename.c_str());
    return 1;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string prompt = buffer.str();

  auto is_loaded = zinferlm::Model::load(model_path.c_str());
  if (!is_loaded)
  {
    LOG_ERR("Failed to load model: %s\n", model_path.c_str());
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

  zinferlm::Model::unload();
  return 0;
}
