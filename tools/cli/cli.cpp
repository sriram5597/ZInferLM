#include <csignal>
#include <iostream>
#include <string>
#include <vector>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

static void sigint_handler(int)
{
  std::cout << "\n";
  _exit(0);
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <model_path>\n";
    return 1;
  }

  std::string model_path = argv[1];

  auto is_loaded = zinferlm::Model::load(model_path.c_str());
  if (!is_loaded)
  {
    std::cerr << "Failed to load model: " << model_path << "\n";
    return 1;
  }

  auto &model = zinferlm::Model::instance();

  zinferlm::model_info_t m_info = model.info();
  std::cout << "Name: " << m_info.name << std::endl;
  std::cout << "Version: " << m_info.version << std::endl;
  std::cout << "Architecture: " << m_info.architecture << std::endl;
  std::cout << "File Type: " << m_info.file_type << std::endl;

  std::signal(SIGINT, sigint_handler);

  std::cout << "\nEnter a prompt (Ctrl+C to exit):\n";
  std::string input;
  while (true)
  {
    std::cout << "> " << std::flush;
    if (!std::getline(std::cin, input))
      break;
    if (input.empty())
      continue;
    if (input[0] == '/')
    {
      size_t space = input.find(' ');
      std::string cmd = input.substr(0, space);
      std::string args = (space != std::string::npos) ? input.substr(space + 1) : "";
      if (cmd == "/tokenize")
      {
        zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(model);
        auto tokens = tokenizer.tokenize(args);
        for (auto &t : tokens)
          std::cout << t << " ";
        std::cout << std::endl;
      }
      else if (cmd == "/graph")
      {
        model.summary();
      }
      else
      {
        std::cout << "Unknown command: " << cmd << std::endl;
      }
      continue;
    }
    std::string output = model.invoke(input, 256);
    std::cout << output << std::endl;
  }

  return 0;
}
