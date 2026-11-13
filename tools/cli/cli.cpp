#include <zinferlm/chat.h>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <vector>
#include <zinferlm/metrics.h>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

static void sigint_handler(int) {
  std::cout << "\n";
  _exit(0);
}

static std::vector<std::string> split(const std::string &s,
                                      const std::string &delim) {
  std::vector<std::string> parts;
  size_t start = 0, end;
  while ((end = s.find(delim, start)) != std::string::npos) {
    parts.push_back(s.substr(start, end - start));
    start = end + delim.size();
  }
  parts.push_back(s.substr(start));
  return parts;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path>\n";
    return 1;
  }

  std::string model_path = argv[1];

  auto is_loaded = zinferlm::Model::load(model_path.c_str());
  if (!is_loaded) {
    std::cerr << "Failed to load model: " << model_path << "\n";
    return 1;
  }

  auto &model = zinferlm::Model::instance();
  zinferlm::model_config_t cfg = model.config();

  zinferlm::model_info_t m_info = model.info();
  std::vector<zinferlm::tensor_info_t> tensors = model.tensor_info();
  std::cout << "Name: " << m_info.name << std::endl;
  std::cout << "Version: " << m_info.version << std::endl;
  std::cout << "Architecture: " << m_info.architecture << std::endl;
  std::cout << "File Type: " << m_info.file_type << std::endl;
  std::cout << "Layers: " << cfg.n_blocks << std::endl;
  std::cout << "Max Context: " << cfg.max_context_len << std::endl;
  std::cout << "Tensors Count: " << tensors.size() << std::endl;

  zinferlm::metrics::LLMMetricsCollector metrics_collector("run-1");
  metrics_collector.collect();

  uint64_t chat_id_counter = 1;

  std::cout << "\nEnter a prompt (Ctrl+C to exit):\n";
  std::string input;
  while (true) {
    std::cout << "> " << std::flush;
    if (!std::getline(std::cin, input))
      break;
    if (input.empty())
      continue;
    if (input[0] == '/') {
      size_t space = input.find(' ');
      std::string cmd = input.substr(0, space);
      std::string args =
          (space != std::string::npos) ? input.substr(space + 1) : "";
      if (cmd == "/tokenize") {
        zinferlm::Tokenizer tokenizer = zinferlm::Tokenizer::for_model(model);
        auto tokens = tokenizer.tokenize(args);
        for (auto &t : tokens)
          std::cout << t << " ";
        std::cout << std::endl;
      } else if (cmd == "/graph") {
        model.summary();
      } else {
        std::cout << "Unknown command: " << cmd << std::endl;
      }
      continue;
    }
    std::vector<std::string> prompts = split(input, ";;;");

    std::vector<std::future<std::string>> futures;
    std::vector<std::string> chat_ids;
    futures.reserve(prompts.size());
    chat_ids.reserve(prompts.size());

    for (const auto &prompt : prompts) {
      if (prompt.empty())
        continue;

      zinferlm::Chat chat(model);
      std::vector<zinferlm::ChatMessage> messages{zinferlm::UserMessage(prompt)};
      const zinferlm::chat_params_t params = {
          .chat_id = std::to_string(chat_id_counter),
          .max_tokens = 8192,
          .temperature = 0.5,
          .top_p = 0.7,
      };
      futures.emplace_back(chat.invoke(messages, params));
      chat_ids.emplace_back(std::to_string(chat_id_counter));
      ++chat_id_counter;
    }

    for (size_t i = 0; i < futures.size(); ++i) {
      std::string out = futures[i].get();
      std::cout << "ChatID: " << chat_ids[i] << "\n"
                << out << "\n"
                << "--\n";
    }
    // zinferlm::metrics::model_metrics_t metrics =
    //     metrics_collector.get_metrics();
    // std::cout << "\033[35m"
    //           << "Time-to-Frist-Token (ms): " << metrics.time_to_first_token
    //           << " | Decode Time (ms): " << metrics.decode_time
    //           << " | Total Time (ms): " << metrics.total_time
    //           << " | Token/s: " << metrics.throughput_in_secs
    //           << " | Input Tokens: " << metrics.inp_tokens
    //           << " | Output Tokens: " << metrics.out_tokens << "\033[0m"
    //           << std::endl;
    // metrics_collector.reset();
  }

  zinferlm::Model::unload();
  return 0;
}
