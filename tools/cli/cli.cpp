#include <iostream>
#include <string>
#include <memory>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

void print_usage(const char *prog_name)
{
  std::cerr << "Usage: " << prog_name << " inspect <model_path>\n";
  std::cerr << "       " << prog_name << " tokenize <model_path> [--info]\n";
}

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    print_usage(argv[0]);
    return 1;
  }

  std::string subcommand = argv[1];
  std::string model_path = argv[2];

  if (subcommand == "inspect")
  {
    auto is_loaded = zinferlm::Model::load(model_path.c_str());
    if (!is_loaded)
    {
      std::cerr << "Failed to load model: " << model_path << "\n";
      return 1;
    }
    zinferlm::model_info_t m_info = zinferlm::Model::instance().info();
    std::cout << "Name: " << m_info.name << std::endl;
    std::cout << "Version: " << m_info.version << std::endl;
    std::cout << "Architecture: " << m_info.architecture << std::endl;
    std::cout << "File Type: " << m_info.file_type << std::endl;

    std::cout << "===========Tensor Info============" << std::endl;
    std::vector<zinferlm::tensor_info_t> tinfo = zinferlm::Model::instance().tensor_info();
    for (auto &t : tinfo)
    {
      std::cout << t.name << " " << t.type << " " << t.n_dim << " " << t.dimensions << std::endl;
    }
  }
  else if (subcommand == "tokenize")
  {
    bool show_info = false;
    bool use_stdin = false;
    for (int i = 3; i < argc; ++i)
    {
      std::string arg = argv[i];
      if (arg == "--info")
      {
        show_info = true;
      }
      else if (arg == "--stdin")
      {
        use_stdin = true;
      }
    }

    auto is_loaded = zinferlm::Model::load(model_path.c_str());
    if (!is_loaded)
    {
      std::cerr << "Failed to load model: " << model_path << "\n";
      return 1;
    }

    auto &model = zinferlm::Model::instance();

    if (show_info)
    {
      zinferlm::tokenizer_info_t ti = model.tokenizer_info();
      std::cout << "Pre: " << ti.pre << std::endl;
      std::cout << "Model: " << ti.model << std::endl;
      std::cout << "Vocab size: " << ti.tokens.size() << "\n";
      std::cout << "BOS token id: " << ti.bos_token_id << "\n";
      std::cout << "EOS token id: " << ti.eos_token_id << "\n";
      std::cout << "PAD token id: " << ti.padding_token_id << "\n";
      std::cout << "Add BOS: " << (ti.add_bos_token ? "true" : "false") << "\n";
    }
    else
    {
      zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(model);
      std::string input;
      if (use_stdin)
      {
        std::getline(std::cin, input);
      }
      std::vector<zinferlm::token_t> tokens = tokenizer.tokenize(input);
      for (auto c : tokens)
      {
        std::cout << c.token << " -> " << c.token_id << std::endl;
      }
      return 1;
    }
  }
  else
  {
    std::cerr << "Unknown subcommand: " << subcommand << "\n";
    print_usage(argv[0]);
    return 1;
  }

  return 0;
}
