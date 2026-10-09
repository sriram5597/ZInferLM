#include "zinferlm/chat.h"
#include "zinferlm/events.h"
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <zinferlm/metrics.h>
#include <zinferlm/models.h>
#include <zinferlm/tokenizer.h>

std::function<void()> cleanup;

static void sigint_handler(int) {
  std::cout << "\n";
  if (cleanup) {
    cleanup();
  }
  _exit(0);
}

using InferenceEvent = zinferlm::events::InferenceEvent;

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

  zinferlm::model_info_t m_info = model.info();
  std::cout << "Name: " << m_info.name << std::endl;
  std::cout << "Version: " << m_info.version << std::endl;
  std::cout << "Architecture: " << m_info.architecture << std::endl;
  std::cout << "File Type: " << m_info.file_type << std::endl;

  zinferlm::metrics::LLMMetricsCollector metrics_collector;
  metrics_collector.collect();

  zinferlm::events::event_callback_t cb =
      [](zinferlm::events::inference_event_t event) -> void {
    if (auto generated_event =
            std::get_if<zinferlm::events::token_generated_event_t>(&event)) {
      std::cout << "\033[33m" << generated_event->token;
      std::cout.flush();
    }
    if (auto completed_event =
            std::get_if<zinferlm::events::generation_completed_event_t>(
                &event)) {
      zinferlm::StreamStatus status = completed_event->status;
      if (status == zinferlm::StreamStatus::MAX_CTX_REACHED) {
        std::cout << "\nMaximum Context Reached\n";
        std::cout.flush();
      }
      if (status == zinferlm::StreamStatus::EOS) {
        std::cout << std::endl;
        std::cout.flush();
      }
      std::cout << "\033[0m";
    }
  };

  zinferlm::events::EventDispatcher &dispatcher =
      zinferlm::events::EventDispatcher::get_instance();

  std::set<InferenceEvent> events = {InferenceEvent::GENERATION_COMPLETED,
                                     InferenceEvent::TOKEN_GENERATED};
  int observer_id = dispatcher.listen(cb, events);
  cleanup = [&dispatcher, observer_id]() -> void {
    dispatcher.unlisten(observer_id);
  };

  std::vector<zinferlm::ChatMessage> messages;

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
    // model.set_stream(handler);
    zinferlm::Chat chat(model);
    zinferlm::UserMessage user_message = zinferlm::UserMessage(input);
    messages.push_back(user_message);
    zinferlm::ChatMessage out_msg = chat.invoke(messages);
    messages.push_back(out_msg);
    zinferlm::metrics::model_metrics_t metrics =
        metrics_collector.get_metrics();
    std::cout << "\033[35m"
              << "Time-to-Frist-Token (ms): " << metrics.time_to_first_token
              << " | Decode Time (ms): " << metrics.decode_time
              << " | Total Time (ms): " << metrics.total_time
              << " | Token/s: " << metrics.throughput_in_secs
              << " | Input Tokens: " << metrics.inp_tokens
              << " | Output Tokens: " << metrics.out_tokens << "\033[0m"
              << std::endl;
    metrics_collector.reset();
  }

  return 0;
}
